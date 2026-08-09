"""Unified, non-authorizing readiness audit for private decision support.

The report separates system evidence from event-specific authorization.  It can
identify missing or stale inputs, but it can never authorize a wager by itself.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from lol_kills.live_totals_candidate import (
    DEVELOPMENT_CANDIDATE_LOCATOR,
    LiveTotalsCandidateError,
    validate_development_candidate,
)
from lol_kills.live_totals_model import FRESHNESS_LIMIT_DAYS, MIN_PATCH_TEST_GAMES
from lol_kills.market_decision import SCHEMA_VERSION as MARKET_AUTHORITY_SCHEMA_VERSION
from lol_kills.v2.evaluation.checks import ValidationFailure
from lol_kills.v2.evaluation.contract_validation import (
    validate_current_contract_validation_inputs,
)
from lol_kills.v2.evaluation.contract_reconciliation_v1 import (
    DEFAULT_OUTPUT as CONTRACT_RECONCILIATION_CANDIDATE_LOCATOR,
    ContractReconciliationError,
    validate_contract_reconciliation_candidate_v1,
)
from lol_kills.v2.evaluation.contract_reconciliation_review_v1 import (
    EXTERNAL_SHA256_ENV as CONTRACT_RECONCILIATION_REVIEW_ENV,
    REGISTRY_LOCATOR as CONTRACT_RECONCILIATION_REVIEW_LOCATOR,
    ContractReconciliationReviewError,
    load_pinned_contract_reconciliation_review_v1,
)
from lol_kills.v2.evaluation.contract_prior_tree_recovery_v1 import (
    DEFAULT_MANIFEST as CONTRACT_PRIOR_TREE_RECOVERY_LOCATOR,
    ContractPriorTreeRecoveryError,
    load_prior_tree_recovery_v1,
)
from lol_kills.v2.market.event_probability_v2 import (
    RECEIPT_SCHEMA_VERSION as EVENT_PROBABILITY_RECEIPT_SCHEMA_VERSION,
)
from lol_kills.v2.market.event_probability_registry_v2 import (
    SCHEMA_VERSION as EVENT_PROBABILITY_REGISTRY_SCHEMA_VERSION,
    EventProbabilityRegistryV2Error,
    expected_entries as expected_event_probability_entries,
    load_pinned_event_probability_registry_v2,
)
from lol_kills.v2.market.betano_br_quote_adapter_candidate_registry_v1 import (
    REGISTERED_CANDIDATE_ARTIFACT_SHA256 as BETANO_QUOTE_ADAPTER_CANDIDATE_ARTIFACT_SHA256,
    REGISTERED_CANDIDATE_LOCATOR as BETANO_QUOTE_ADAPTER_CANDIDATE_LOCATOR,
    REGISTERED_CANDIDATE_RAW_SHA256 as BETANO_QUOTE_ADAPTER_CANDIDATE_RAW_SHA256,
    BetanoQuoteAdapterCandidateRegistryError,
    validate_registered_betano_quote_adapter_candidate_v1,
)
from lol_kills.v2.market.betano_br_quote_adapter_registry_v1 import (
    BetanoQuoteAdapterRegistryError,
    load_registered_betano_quote_adapter_v1,
)
from lol_kills.v2.market.betano_br_quote_qualification_v1 import (
    SCHEMA_VERSION as BETANO_QUOTE_QUALIFICATION_SCHEMA_VERSION,
)
from lol_kills.v2.market.betano_br_quote_registry_v2 import (
    EXTERNAL_SHA256_ENV as BETANO_QUOTE_REGISTRY_ENV,
    REGISTRY_LOCATOR as BETANO_QUOTE_REGISTRY_LOCATOR,
    SCHEMA_VERSION as BETANO_QUOTE_REGISTRY_SCHEMA_VERSION,
    BetanoQuoteRegistryV2Error,
    expected_entries as expected_betano_quote_entries,
    load_pinned_betano_quote_registry_v2,
)
from lol_kills.v2.market.betano_terms_snapshot_registry_v1 import (
    REGISTERED_SNAPSHOT_ARTIFACT_SHA256 as BETANO_TERMS_SNAPSHOT_ARTIFACT_SHA256,
    REGISTERED_SNAPSHOT_LOCATOR as BETANO_TERMS_SNAPSHOT_LOCATOR,
    REGISTERED_SNAPSHOT_RAW_SHA256 as BETANO_TERMS_SNAPSHOT_RAW_SHA256,
    BetanoTermsSnapshotRegistryError,
    validate_registered_betano_terms_snapshot_v1,
)
from lol_kills.v2.market.betano_terms_authority_v1 import (
    BetanoTermsAuthorityError,
    load_pinned_betano_terms_authority_v1,
)
from lol_kills.v2.market.calibration_uncertainty_registry_v1 import (
    CalibrationUncertaintyRegistryError,
    expected_registration_binding as expected_calibration_uncertainty_binding,
    load_pinned_calibration_uncertainty_registry,
)
from lol_kills.v2.market.match_winner_future_protocol_registry_v1 import (
    REGISTERED_PROTOCOL_ARTIFACT_SHA256 as MATCH_WINNER_PROTOCOL_ARTIFACT_SHA256,
    REGISTERED_PROTOCOL_LOCATOR as MATCH_WINNER_PROTOCOL_LOCATOR,
    REGISTERED_PROTOCOL_RAW_SHA256 as MATCH_WINNER_PROTOCOL_RAW_SHA256,
    REGISTERED_QUOTE_CAPTURE_CONTRACT_SHA256,
    REGISTERED_SETTLEMENT_CONTRACT_SHA256,
    MatchWinnerFutureProtocolRegistryError,
    validate_registered_match_winner_future_protocol_v1,
)
from lol_kills.v2.market.phase_one_collection_v1 import (
    BUNDLE_PREFIX as PHASE_ONE_BUNDLE_PREFIX,
    BUNDLE_SCHEMA_VERSION as PHASE_ONE_BUNDLE_SCHEMA_VERSION,
    PLAN_PREFIX as PHASE_ONE_PLAN_PREFIX,
    PLAN_SCHEMA_VERSION as PHASE_ONE_PLAN_SCHEMA_VERSION,
    SNAPSHOT_PREFIX as PHASE_ONE_SNAPSHOT_PREFIX,
    SNAPSHOT_SCHEMA_VERSION as PHASE_ONE_SNAPSHOT_SCHEMA_VERSION,
)
from lol_kills.v2.market.prospective_capture_v1 import (
    ATTEMPT_PREFIX as PROSPECTIVE_CAPTURE_ATTEMPT_PREFIX,
    SOURCE_LOCATOR as PROSPECTIVE_CAPTURE_SOURCE_LOCATOR,
    ProspectiveCaptureError,
    validate_attempt_receipt as validate_prospective_capture_attempt,
)
from lol_kills.v2.ratings.player.pre_side_rating_envelope_v1 import (
    ENVELOPE_PREFIX as PRE_SIDE_ENVELOPE_PREFIX,
    SOURCE_LOCATOR as PRE_SIDE_ENVELOPE_SOURCE_LOCATOR,
    PreSideRatingEnvelopeError,
    validate_pre_side_rating_envelope,
)
from lol_kills.v2.ratings.player.pre_side_rating_binding_v1 import (
    BINDING_PREFIX as PRE_SIDE_BINDING_PREFIX,
    SOURCE_LOCATOR as PRE_SIDE_BINDING_SOURCE_LOCATOR,
    PreSideRatingBindingError,
    validate_pre_side_rating_binding,
)
from lol_kills.v2.ratings.player.multileague_v3_side_neutral_protocol_registry_v2 import (
    REGISTERED_PROTOCOL_ARTIFACT_SHA256 as SIDE_NEUTRAL_PROTOCOL_ARTIFACT_SHA256,
    REGISTERED_PROTOCOL_LOCATOR as SIDE_NEUTRAL_PROTOCOL_LOCATOR,
    REGISTERED_PROTOCOL_RAW_SHA256 as SIDE_NEUTRAL_PROTOCOL_RAW_SHA256,
    SideNeutralProtocolRegistryV2Error,
    validate_registered_side_neutral_protocol_v2,
)
from lol_kills.v2.ratings.player.side_neutral_protocol_review_v1 import (
    REVIEW_LOCATOR as SIDE_NEUTRAL_REVIEW_LOCATOR,
    SideNeutralProtocolReviewError,
    load_active_side_neutral_protocol_review,
)
from lol_kills.v2.ratings.player.side_neutral_collection_implementation_registry_v1 import (
    SideNeutralCollectionImplementationRegistryError,
    validate_registered_side_neutral_collection_implementation,
)
from lol_kills.v2.ratings.player.side_neutral_review_packet_v1 import (
    DEFAULT_OUTPUT as SIDE_NEUTRAL_REVIEW_PACKET_LOCATOR,
    SideNeutralReviewPacketError,
    validate_side_neutral_review_packet,
)
from lol_kills.v2.draft.terminal.side_neutral_prediction_v1 import (
    PREDICTION_PREFIX as SIDE_NEUTRAL_DRAFT_PREFIX,
    SOURCE_LOCATOR as SIDE_NEUTRAL_DRAFT_SOURCE_LOCATOR,
    SideNeutralDraftPredictionError,
    validate_side_neutral_draft_prediction,
)
from lol_kills.v2.market.side_neutral_capture_bundle_v1 import (
    BUNDLE_PREFIX as SIDE_NEUTRAL_BUNDLE_PREFIX,
    SOURCE_LOCATOR as SIDE_NEUTRAL_BUNDLE_SOURCE_LOCATOR,
    SideNeutralCaptureBundleError,
    validate_side_neutral_capture_bundle,
)
from lol_kills.v2.market.side_neutral_prospective_capture_v1 import (
    PHASE_ONE_BRIDGE_STAGES as SIDE_NEUTRAL_PHASE_ONE_BRIDGE_STAGES,
    SOURCE_LOCATOR as SIDE_NEUTRAL_OPERATOR_SOURCE_LOCATOR,
    STAGES as SIDE_NEUTRAL_OPERATOR_STAGES,
)
from lol_kills.v2.market.side_neutral_ledger_v1 import (
    DEFAULT_LEDGER as SIDE_NEUTRAL_LEDGER_LOCATOR,
    SideNeutralLedgerError,
    validate_side_neutral_ledger,
)
from lol_kills.v2.market.phase_one_collection_readiness_registry_v1 import (
    REGISTERED_READINESS_ARTIFACT_SHA256 as PHASE_ONE_READINESS_ARTIFACT_SHA256,
    REGISTERED_READINESS_LOCATOR as PHASE_ONE_READINESS_LOCATOR,
    REGISTERED_READINESS_RAW_SHA256 as PHASE_ONE_READINESS_RAW_SHA256,
    PhaseOneCollectionReadinessRegistryError,
    validate_registered_phase_one_collection_readiness_v1,
)
from lol_kills.v2.market.phase_one_evaluation_readiness_registry_v1 import (
    REGISTERED_READINESS_ARTIFACT_SHA256 as PHASE_ONE_EVALUATION_READINESS_ARTIFACT_SHA256,
    REGISTERED_READINESS_LOCATOR as PHASE_ONE_EVALUATION_READINESS_LOCATOR,
    REGISTERED_READINESS_RAW_SHA256 as PHASE_ONE_EVALUATION_READINESS_RAW_SHA256,
    RegisteredPhaseOneEvaluationReadinessError,
    validate_registered_phase_one_evaluation_readiness_v1,
)
from lol_kills.v2.market.phase_one_evaluation_registry_v1 import (
    EXTERNAL_SHA256_ENV as PHASE_ONE_EVALUATION_REGISTRY_ENV,
    REGISTRY_LOCATOR as PHASE_ONE_EVALUATION_REGISTRY_LOCATOR,
    PhaseOneEvaluationRegistryError,
    expected_result_binding,
    load_pinned_evaluation_registry,
)
from lol_kills.v2.market import phase_one_evaluation_v1 as phase_one_model_evaluation
from lol_kills.v2.market.probability_pipeline_readiness_registry_v1 import (
    REGISTERED_READINESS_ARTIFACT_SHA256 as PROBABILITY_PIPELINE_READINESS_ARTIFACT_SHA256,
    REGISTERED_READINESS_LOCATOR as PROBABILITY_PIPELINE_READINESS_LOCATOR,
    REGISTERED_READINESS_RAW_SHA256 as PROBABILITY_PIPELINE_READINESS_RAW_SHA256,
    RegisteredProbabilityPipelineReadinessError,
    validate_registered_probability_pipeline_readiness_v1,
)
from lol_kills.v2.market.phase_two_opening_v1 import (
    PhaseTwoOpeningError,
    validate_active_phase_two_opening,
)
from lol_kills.v2.market.phase_two_collection_readiness_registry_v1 import (
    EXTERNAL_SHA256_ENV as PHASE_TWO_COLLECTION_READINESS_ENV,
    REGISTRY_LOCATOR as PHASE_TWO_COLLECTION_READINESS_REGISTRY_LOCATOR,
    PhaseTwoCollectionReadinessRegistryError,
    expected_readiness_binding as expected_phase_two_collection_readiness_binding,
    load_pinned_phase_two_collection_readiness_registry_v1,
)
from lol_kills.v2.market.phase_two_evaluation_readiness_registry_v1 import (
    EXTERNAL_SHA256_ENV as PHASE_TWO_EVALUATION_READINESS_ENV,
    REGISTRY_LOCATOR as PHASE_TWO_EVALUATION_READINESS_REGISTRY_LOCATOR,
    PhaseTwoEvaluationReadinessRegistryError,
    expected_readiness_binding as expected_phase_two_evaluation_readiness_binding,
    load_pinned_phase_two_evaluation_readiness_registry_v1,
)
from lol_kills.v2.market.phase_two_stopping_snapshot_registry_v1 import (
    EXTERNAL_SHA256_ENV as PHASE_TWO_SNAPSHOT_REGISTRY_ENV,
    REGISTRY_LOCATOR as PHASE_TWO_SNAPSHOT_REGISTRY_LOCATOR,
    PhaseTwoSnapshotRegistryError,
    expected_snapshot_binding as expected_phase_two_snapshot_binding,
    load_pinned_phase_two_snapshot_registry_v1,
)
from lol_kills.v2.market.phase_two_evaluation_result_registry_v1 import (
    EXTERNAL_SHA256_ENV as PHASE_TWO_EVALUATION_REGISTRY_ENV,
    REGISTRY_LOCATOR as PHASE_TWO_EVALUATION_REGISTRY_LOCATOR,
    PhaseTwoEvaluationRegistryError,
    expected_result_binding as expected_phase_two_result_binding,
    load_pinned_phase_two_evaluation_registry_v1,
)
from lol_kills.v2.market.semantic_market_authority_v1 import (
    AUTHORITY_LOCATOR as SEMANTIC_MATCH_WINNER_AUTHORITY_LOCATOR,
    EXTERNAL_SHA256_ENV as SEMANTIC_MATCH_WINNER_AUTHORITY_ENV,
    SemanticMarketAuthorityError,
    load_active_semantic_market_authority_v1,
)
from lol_kills.v2.draft.terminal.l2_readiness import inspect_l2_readiness
from lol_kills.v2.draft.terminal.semantic_draft_authority_v1 import (
    AUTHORITY_LOCATOR as SEMANTIC_DRAFT_AUTHORITY_LOCATOR,
    EXTERNAL_SHA256_ENV as SEMANTIC_DRAFT_AUTHORITY_ENV,
)
from lol_kills.v2.ratings.player.multileague_development import (
    DEFAULT_MAPS_LOCATOR,
    DEFAULT_PLAYERS_LOCATOR,
)
from lol_kills.v2.ratings.player.multileague_runner import (
    MultiLeagueRunnerError,
    validate_multileague_development_artifact,
)
from lol_kills.v2.ratings.player.multileague_benchmark import (
    MultiLeagueBenchmarkError,
    validate_strong_baseline_benchmark,
)
from lol_kills.v2.ratings.player.multileague_v2_protocol_equal_series import (
    EqualSeriesProtocolError,
    validate_equal_series_protocol_lock,
)
from lol_kills.v2.ratings.player.multileague_v2_runner_equal_series import (
    EqualSeriesRunnerError,
    validate_equal_series_adaptive_artifact,
)
from lol_kills.v2.ratings.player.multileague_v2_sealed_authority import (
    inspect_sealed_opening_authority,
)
from lol_kills.v2.ratings.player.multileague_source_snapshot import (
    CURRENT_MANIFEST_LOCATOR as V3_SOURCE_MANIFEST_V1_LOCATOR,
    MultiLeagueSourceSnapshotError,
    validate_current_source_snapshot,
)
from lol_kills.v2.ratings.player.multileague_v3_registry import (
    FutureProtocolRegistryError as FutureProtocolRegistryV1Error,
    REGISTERED_PROTOCOL_LOCATOR as V3_PROTOCOL_V1_LOCATOR,
    validate_registered_future_protocol as validate_registered_future_protocol_v1,
)
from lol_kills.v2.ratings.player.multileague_v3_source_registry_v2 import (
    MANIFEST_LOCATOR as V3_SOURCE_MANIFEST_V2_LOCATOR,
    SourceRegistryV2Error,
    validate_registered_source_snapshot_v2,
)
from lol_kills.v2.ratings.player.multileague_v3_preflight_v1_registry import (
    REGISTERED_PREFLIGHT_LOCATOR as V3_PREFLIGHT_V1_LOCATOR,
    SourcePreflightRegistryV1Error,
    validate_registered_source_preflight_v1,
)
from lol_kills.v2.ratings.player.multileague_v3_preflight_v2_registry import (
    REGISTERED_PREFLIGHT_LOCATOR as V3_PREFLIGHT_V2_REJECTED_LOCATOR,
    SourcePreflightRegistryV2Error,
    validate_registered_source_preflight_v2,
)
from lol_kills.v2.ratings.player.multileague_v3_preflight_v3_registry import (
    REGISTERED_PREFLIGHT_LOCATOR as V3_PREFLIGHT_V3_LOCATOR,
    SourcePreflightRegistryV3Error,
    validate_registered_source_preflight_v3,
)
from lol_kills.v2.ratings.player.multileague_v3_registry_v2 import (
    FutureProtocolRegistryV2Error,
    REGISTERED_PROTOCOL_LOCATOR as V3_PROTOCOL_V2_REJECTED_LOCATOR,
    validate_registered_future_protocol_v2,
)
from lol_kills.v2.ratings.player.multileague_v3_registry_v3 import (
    FutureProtocolRegistryV3Error,
    REGISTERED_PROTOCOL_LOCATOR as V3_PROTOCOL_V3_LOCATOR,
    validate_registered_future_protocol_v3,
)
from lol_kills.v2.ratings.player.multileague_v3_corrected_adaptive_diagnostic_registry_v1 import (
    CorrectedAdaptiveDiagnosticRegistryError,
    REGISTERED_DIAGNOSTIC_LOCATOR as V3_CORRECTED_ADAPTIVE_DIAGNOSTIC_LOCATOR,
    validate_registered_corrected_adaptive_diagnostic_v1,
)
from lol_kills.v2.ratings.player.multileague_v3_capture_registry import (
    CaptureReadinessRegistryError as CaptureReadinessRegistryV1Error,
    REGISTERED_CAPTURE_LOCATOR as V3_CAPTURE_READINESS_V1_REJECTED_LOCATOR,
    validate_registered_capture_readiness as validate_registered_capture_readiness_v1,
)
from lol_kills.v2.ratings.player.multileague_v3_capture_registry_v2 import (
    CaptureReadinessRegistryV2Error,
    REGISTERED_CAPTURE_LOCATOR as V3_CAPTURE_READINESS_V2_SUPERSEDED_LOCATOR,
    validate_registered_capture_readiness_v2,
)
from lol_kills.v2.ratings.player.multileague_v3_capture_registry_v3 import (
    CaptureReadinessRegistryV3Error,
    REGISTERED_CAPTURE_LOCATOR as V3_CAPTURE_READINESS_V3_LOCATOR,
    validate_registered_capture_readiness_v3,
)
from lol_kills.v2.ratings.player.multileague_v3_prediction_ledger import (
    DEFAULT_REGISTRY as V3_PREDICTION_LEDGER_LOCATOR,
    PredictionLedgerError as V3PredictionLedgerError,
    validate_prediction_ledger_registry as validate_v3_prediction_ledger,
)
from lol_kills.v2.ratings.player.multileague_v3_temporal_failure_registry import (
    REGISTERED_FAILURE_LOCATOR as V3_TEMPORAL_FAILURE_LOCATOR,
    TemporalFailureRegistryError,
    validate_registered_temporal_failure,
)
from lol_kills.v2.ratings.semantic_rating_authority_v1 import (
    AUTHORITY_LOCATOR as SEMANTIC_RATING_AUTHORITY_LOCATOR,
    EXTERNAL_SHA256_ENV as SEMANTIC_RATING_AUTHORITY_ENV,
    SemanticRatingAuthorityError,
    load_active_semantic_rating_authority_v1,
)


SCHEMA_VERSION = "scryglass.private-decision-readiness.v15"
PLAYER_ARTIFACT = Path(
    "data/lol/v2/models/player/real-v1/private-development-artifact-v3.json"
)
TEAM_ARTIFACT = Path(
    "data/lol/v2/models/team/real-v1/private-development-artifact-v3.json"
)
MULTILEAGUE_RATING_ARTIFACT = Path(
    "data/lol/v2/models/player/multileague-v1/private-development-artifact-v1.json"
)
STRONG_BASELINE_ARTIFACT = Path(
    "data/lol/v2/models/player/multileague-v1/strong-baseline-benchmark-v1.json"
)
V2_RATING_PROTOCOL_ARTIFACT = Path(
    "data/lol/v2/models/player/multileague-v2/protocol-lock-v2.json"
)
V2_RATING_SELECTION_ARTIFACT = Path(
    "data/lol/v2/models/player/multileague-v2/adaptive-development-artifact-v2.json"
)
LIVE_TOTALS_ARTIFACT = DEVELOPMENT_CANDIDATE_LOCATOR

REGISTRATIONS = {
    "match_winner_market_authority": (
        SEMANTIC_MATCH_WINNER_AUTHORITY_LOCATOR,
        SEMANTIC_MATCH_WINNER_AUTHORITY_ENV,
    ),
    "total_kills_market_authority": (
        Path("data/lol/private_market_authority/total_kills.json"),
        "SCRYGLASS_PRIVATE_TOTAL_KILLS_AUTHORITY_SHA256",
    ),
    "quote_registry": (
        Path("data/lol/private_market_quotes/registry.json"),
        "SCRYGLASS_PRIVATE_QUOTE_REGISTRY_SHA256",
    ),
    "roster_registry": (
        Path("data/lol/private_pregame_rosters/registry.json"),
        "SCRYGLASS_PRIVATE_ROSTER_REGISTRY_SHA256",
    ),
    "rating_registry": (
        Path("data/lol/private_rating_authority/registry.json"),
        "SCRYGLASS_PRIVATE_RATING_REGISTRY_SHA256",
    ),
    "semantic_rating_authority": (
        SEMANTIC_RATING_AUTHORITY_LOCATOR,
        SEMANTIC_RATING_AUTHORITY_ENV,
    ),
    "semantic_draft_authority": (
        SEMANTIC_DRAFT_AUTHORITY_LOCATOR,
        SEMANTIC_DRAFT_AUTHORITY_ENV,
    ),
    "match_winner_phase_one_evaluation": (
        PHASE_ONE_EVALUATION_REGISTRY_LOCATOR,
        PHASE_ONE_EVALUATION_REGISTRY_ENV,
    ),
    "match_winner_calibration_uncertainty": (
        Path(
            "data/lol/private_market_authority/"
            "phase-one-recalibration-uncertainty-registry-v1.json"
        ),
        "SCRYGLASS_PRIVATE_MATCH_WINNER_CALIBRATION_SHA256",
    ),
    "match_winner_bookmaker_terms": (
        Path(
            "data/lol/v2/evaluation/match-winner-market-v1/bookmaker-terms-registry.json"
        ),
        "SCRYGLASS_PRIVATE_MATCH_WINNER_BOOKMAKER_TERMS_SHA256",
    ),
    "match_winner_quote_adapter": (
        Path(
            "data/lol/v2/evaluation/match-winner-market-v1/quote-adapter-registry.json"
        ),
        "SCRYGLASS_PRIVATE_MATCH_WINNER_QUOTE_ADAPTER_SHA256",
    ),
    "match_winner_event_probability_registry": (
        Path(
            "data/lol/v2/evaluation/match-winner-market-v1/event-probability-registry.json"
        ),
        "SCRYGLASS_PRIVATE_MATCH_WINNER_PROBABILITY_REGISTRY_SHA256",
    ),
    "match_winner_quote_registry": (
        BETANO_QUOTE_REGISTRY_LOCATOR,
        BETANO_QUOTE_REGISTRY_ENV,
    ),
    "match_winner_phase_two_opening": (
        Path(
            "data/lol/v2/evaluation/match-winner-market-v1/phase-two-opening-authority.json"
        ),
        "SCRYGLASS_PRIVATE_MATCH_WINNER_PHASE_TWO_OPENING_SHA256",
    ),
    "match_winner_phase_two_collection_readiness": (
        PHASE_TWO_COLLECTION_READINESS_REGISTRY_LOCATOR,
        PHASE_TWO_COLLECTION_READINESS_ENV,
    ),
    "match_winner_phase_two_evaluation_readiness": (
        PHASE_TWO_EVALUATION_READINESS_REGISTRY_LOCATOR,
        PHASE_TWO_EVALUATION_READINESS_ENV,
    ),
    "match_winner_phase_two_stopping_snapshot": (
        PHASE_TWO_SNAPSHOT_REGISTRY_LOCATOR,
        PHASE_TWO_SNAPSHOT_REGISTRY_ENV,
    ),
    "match_winner_phase_two_evaluation": (
        PHASE_TWO_EVALUATION_REGISTRY_LOCATOR,
        PHASE_TWO_EVALUATION_REGISTRY_ENV,
    ),
}


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(root: Path, locator: Path) -> dict[str, Any]:
    path = root / locator
    result: dict[str, Any] = {
        "locator": locator.as_posix(),
        "present": path.is_file(),
        "raw_sha256": None,
        "payload": None,
        "error": None,
    }
    if not path.is_file():
        result["error"] = "artifact_missing"
        return result
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("artifact root must be an object")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        result["error"] = f"artifact_invalid:{exc}"
        return result
    result["raw_sha256"] = hashlib.sha256(raw).hexdigest()
    result["payload"] = payload
    return result


def _rating_readiness(
    root: Path,
    environment: Mapping[str, str],
    as_of: datetime,
) -> dict[str, Any]:
    multileague_record = _read_json(root, MULTILEAGUE_RATING_ARTIFACT)
    multileague = multileague_record.get("payload") or {}
    if multileague:
        try:
            multileague = validate_multileague_development_artifact(multileague)
        except (MultiLeagueRunnerError, OSError, ValueError) as exc:
            multileague_record["error"] = f"artifact_semantics_invalid:{exc}"
            multileague_record["payload"] = None
            multileague = {}
    legacy_player_record = _read_json(root, PLAYER_ARTIFACT)
    legacy_team_record = _read_json(root, TEAM_ARTIFACT)
    legacy_player = legacy_player_record.get("payload") or {}
    legacy_team = legacy_team_record.get("payload") or {}
    benchmark_record = _read_json(root, STRONG_BASELINE_ARTIFACT)
    benchmark = benchmark_record.get("payload") or {}
    if benchmark:
        try:
            benchmark = validate_strong_baseline_benchmark(benchmark)
        except (MultiLeagueBenchmarkError, OSError, ValueError) as exc:
            benchmark_record["error"] = f"artifact_semantics_invalid:{exc}"
            benchmark_record["payload"] = None
            benchmark = {}
    protocol_record = _read_json(root, V2_RATING_PROTOCOL_ARTIFACT)
    protocol_raw = protocol_record.get("payload") or {}
    protocol = protocol_raw
    protocol_integrity_without_source_replay = False
    if protocol:
        try:
            protocol = validate_equal_series_protocol_lock(protocol, root=root)
        except (EqualSeriesProtocolError, OSError, ValueError) as exc:
            protocol_integrity_without_source_replay = str(exc).startswith(
                "bound source drifted:"
            )
            protocol_record["error"] = f"artifact_semantics_invalid:{exc}"
            protocol_record["payload"] = None
            protocol = {}
    v2_selection_record = _read_json(root, V2_RATING_SELECTION_ARTIFACT)
    v2_selection_raw = v2_selection_record.get("payload") or {}
    v2_selection = v2_selection_raw
    selection_integrity_without_source_replay = False
    if v2_selection:
        try:
            v2_selection = validate_equal_series_adaptive_artifact(
                v2_selection,
                root=root,
            )
        except (EqualSeriesRunnerError, OSError, ValueError) as exc:
            selection_integrity_without_source_replay = str(exc).startswith(
                "bound source drifted:"
            )
            v2_selection_record["error"] = f"artifact_semantics_invalid:{exc}"
            v2_selection_record["payload"] = None
            v2_selection = {}
    protocol_view = (
        protocol_raw if protocol_integrity_without_source_replay else protocol
    )
    selection_view = (
        v2_selection_raw
        if selection_integrity_without_source_replay
        else v2_selection
    )
    protocol_boundary = protocol_view.get("information_boundary") or {}
    protocol_disclosure = protocol_view.get("adaptation_disclosure") or {}
    protocol_final_gate = protocol_view.get("sealed_final_gate") or {}
    sealed_opening_authority = inspect_sealed_opening_authority(
        root,
        environment=environment,
    )
    v3_source_snapshot_v1: dict[str, Any] = {}
    v3_source_snapshot_v1_error: str | None = None
    try:
        v3_source_snapshot_v1 = validate_current_source_snapshot(root=root)
    except (MultiLeagueSourceSnapshotError, OSError, ValueError) as exc:
        v3_source_snapshot_v1_error = str(exc)
    v3_protocol_v1: dict[str, Any] = {}
    v3_protocol_v1_error: str | None = None
    try:
        v3_protocol_v1 = validate_registered_future_protocol_v1(root=root)
    except (FutureProtocolRegistryV1Error, OSError, ValueError) as exc:
        v3_protocol_v1_error = str(exc)
    v3_preflight_v1: dict[str, Any] = {}
    v3_preflight_v1_error: str | None = None
    try:
        v3_preflight_v1 = validate_registered_source_preflight_v1(root=root)
    except (SourcePreflightRegistryV1Error, OSError, ValueError) as exc:
        v3_preflight_v1_error = str(exc)
    v3_source_snapshot: dict[str, Any] = {}
    v3_source_snapshot_error: str | None = None
    try:
        v3_source_snapshot = validate_registered_source_snapshot_v2(root=root)
    except (SourceRegistryV2Error, OSError, ValueError) as exc:
        v3_source_snapshot_error = str(exc)
    v3_temporal_failure: dict[str, Any] = {}
    v3_temporal_failure_error: str | None = None
    try:
        v3_temporal_failure = validate_registered_temporal_failure(root=root)
    except (TemporalFailureRegistryError, OSError, ValueError) as exc:
        v3_temporal_failure_error = str(exc)
    v3_temporal_failures = {
        str(item.get("kind")): item
        for item in (v3_temporal_failure.get("failures") or [])
        if isinstance(item, Mapping)
    }
    v3_preflight_v2_rejected: dict[str, Any] = {}
    v3_preflight_v2_rejected_error: str | None = None
    try:
        v3_preflight_v2_rejected = validate_registered_source_preflight_v2(
            root=root
        )
    except (SourcePreflightRegistryV2Error, OSError, ValueError) as exc:
        v3_preflight_v2_rejected_error = str(exc)
    v3_preflight_v3: dict[str, Any] = {}
    v3_preflight_v3_error: str | None = None
    try:
        v3_preflight_v3 = validate_registered_source_preflight_v3(root=root)
    except (SourcePreflightRegistryV3Error, OSError, ValueError) as exc:
        v3_preflight_v3_error = str(exc)
    v3_future_protocol_v2_rejected: dict[str, Any] = {}
    v3_future_protocol_v2_rejected_error: str | None = None
    try:
        v3_future_protocol_v2_rejected = validate_registered_future_protocol_v2(
            root=root
        )
    except (FutureProtocolRegistryV2Error, OSError, ValueError) as exc:
        v3_future_protocol_v2_rejected_error = str(exc)
    v3_future_protocol: dict[str, Any] = {}
    v3_future_protocol_error: str | None = None
    try:
        v3_future_protocol = validate_registered_future_protocol_v3(root=root)
    except (FutureProtocolRegistryV3Error, OSError, ValueError) as exc:
        v3_future_protocol_error = str(exc)
    v3_corrected_adaptive_diagnostic: dict[str, Any] = {}
    v3_corrected_adaptive_diagnostic_error: str | None = None
    try:
        v3_corrected_adaptive_diagnostic = (
            validate_registered_corrected_adaptive_diagnostic_v1(root=root)
        )
    except (
        CorrectedAdaptiveDiagnosticRegistryError,
        OSError,
        ValueError,
    ) as exc:
        v3_corrected_adaptive_diagnostic_error = str(exc)
    v3_future_holdout = v3_future_protocol.get("future_holdout") or {}
    v3_opening_authority = v3_future_protocol.get("opening_authority") or {}
    v3_decision_outputs = v3_future_protocol.get("decision_outputs") or {}
    v3_prediction_ledger = v3_future_protocol.get("prediction_ledger") or {}
    v3_capture_readiness_v1_record = _read_json(
        root, V3_CAPTURE_READINESS_V1_REJECTED_LOCATOR
    )
    v3_capture_readiness_v1_rejected = (
        v3_capture_readiness_v1_record.get("payload") or {}
    )
    v3_capture_readiness_v1_rejected_error: str | None = None
    try:
        validate_registered_capture_readiness_v1(root=root)
    except (CaptureReadinessRegistryV1Error, OSError, ValueError) as exc:
        v3_capture_readiness_v1_rejected_error = str(exc)
    v3_capture_readiness_v2_record = _read_json(
        root, V3_CAPTURE_READINESS_V2_SUPERSEDED_LOCATOR
    )
    v3_capture_readiness_v2_superseded = (
        v3_capture_readiness_v2_record.get("payload") or {}
    )
    v3_capture_readiness_v2_superseded_error: str | None = None
    try:
        validate_registered_capture_readiness_v2(root=root)
    except (CaptureReadinessRegistryV2Error, OSError, ValueError) as exc:
        v3_capture_readiness_v2_superseded_error = str(exc)
    v3_capture_readiness: dict[str, Any] = {}
    v3_capture_readiness_error: str | None = None
    try:
        v3_capture_readiness = validate_registered_capture_readiness_v3(root=root)
    except (CaptureReadinessRegistryV3Error, OSError, ValueError) as exc:
        v3_capture_readiness_error = str(exc)
    v3_capture_contract = v3_capture_readiness.get("capture_contract") or {}
    v3_capture_implementation = v3_capture_readiness.get("implementation") or {}
    v3_capture_ledger_state = v3_capture_readiness.get("ledger_state") or {}
    v3_live_ledger_record = _read_json(root, V3_PREDICTION_LEDGER_LOCATOR)
    v3_live_ledger = v3_live_ledger_record.get("payload") or {}
    v3_live_ledger_error: str | None = None
    if v3_live_ledger:
        try:
            v3_live_ledger = validate_v3_prediction_ledger(
                v3_live_ledger,
                root=root,
            )
        except (V3PredictionLedgerError, OSError, ValueError) as exc:
            v3_live_ledger_error = str(exc)
            v3_live_ledger_record["error"] = f"artifact_semantics_invalid:{exc}"
            v3_live_ledger = {}
    v3_joint_evaluation_registry: dict[str, Any] = {}
    v3_joint_evaluation_result: dict[str, Any] = {}
    v3_joint_evaluation_error: str | None = None
    v3_joint_registry_path = root / PHASE_ONE_EVALUATION_REGISTRY_LOCATOR
    v3_joint_registry_digest = environment.get(PHASE_ONE_EVALUATION_REGISTRY_ENV)
    if v3_joint_registry_path.is_file() and v3_joint_registry_digest:
        try:
            registry_payload = json.loads(v3_joint_registry_path.read_text())
            result_locator = (
                registry_payload.get("result_binding") or {}
            ).get("result_locator")
            if not isinstance(result_locator, str):
                raise PhaseOneEvaluationRegistryError(
                    "joint phase-one result locator is missing"
                )
            binding = expected_result_binding(
                result_locator=result_locator,
                root=root,
            )
            v3_joint_evaluation_registry = load_pinned_evaluation_registry(
                path=v3_joint_registry_path,
                external_sha256=v3_joint_registry_digest,
                expected_binding=binding,
            )
            result_raw = phase_one_model_evaluation._read_regular(
                root,
                result_locator,
                "joint phase-one evaluation result",
            )
            v3_joint_evaluation_result = (
                phase_one_model_evaluation.validate_phase_one_evaluation_result(
                    phase_one_model_evaluation._strict_object(
                        result_raw,
                        "joint phase-one evaluation result",
                    )
                )
            )
        except (
            json.JSONDecodeError,
            PhaseOneEvaluationRegistryError,
            phase_one_model_evaluation.PhaseOneEvaluationError,
            OSError,
            ValueError,
        ) as exc:
            v3_joint_evaluation_error = str(exc)
            v3_joint_evaluation_registry = {}
            v3_joint_evaluation_result = {}

    selection = multileague.get("selection") or {}
    posterior = multileague.get("development_posterior") or {}
    teams = posterior.get("teams") if isinstance(posterior, Mapping) else None
    players = posterior.get("players") if isinstance(posterior, Mapping) else None
    winner_id = selection.get("development_winner_candidate_id")
    winner = next(
        (
            item
            for item in multileague.get("candidate_results", [])
            if isinstance(item, Mapping)
            and (item.get("candidate") or {}).get("candidate_id") == winner_id
        ),
        {},
    )
    validation_comparison = (winner.get("paired_against_static") or {}).get(
        "validation"
    ) or {}
    lcs_comparison = next(
        (
            item
            for item in validation_comparison.get("domestic_leagues", [])
            if isinstance(item, Mapping) and item.get("league") == "LCS"
        ),
        {},
    )
    strong_comparison = benchmark.get(
        "validation_player_minus_selected_organization"
    ) or {}
    strong_lcs = next(
        (
            item
            for item in strong_comparison.get("by_domestic_league", [])
            if isinstance(item, Mapping) and item.get("league") == "LCS"
        ),
        {},
    )
    strong_roster_change = next(
        (
            item
            for item in strong_comparison.get("by_roster_change_stratum", [])
            if isinstance(item, Mapping)
            and item.get("roster_change_stratum")
            == "ONE_OR_BOTH_ROSTERS_CHANGED"
        ),
        {},
    )
    joint_ratings = v3_joint_evaluation_result.get("ratings_evaluation") or {}
    joint_rating_metrics = joint_ratings.get("metrics_by_stratum") or {}
    joint_rating_comparators = joint_ratings.get("comparators") or []

    def joint_rating_stratum_passed(stratum: str) -> bool:
        report = joint_rating_metrics.get(stratum) or {}
        return bool(joint_rating_comparators) and all(
            (report.get(comparator) or {}).get(metric, {}).get("upper_95")
            is not None
            and (report.get(comparator) or {}).get(metric, {}).get("upper_95")
            <= 0.0
            for comparator in joint_rating_comparators
            for metric in ("log_loss", "brier")
        )

    joint_models_passed = (
        v3_joint_evaluation_registry.get(
            "phase_one_evaluation_independently_registered"
        )
        is True
        and v3_joint_evaluation_registry.get(
            "phase_one_models_independently_passed"
        )
        is True
        and v3_joint_evaluation_result.get("phase_one_models_passed") is True
    )
    semantic_rating_authority: dict[str, Any] = {}
    semantic_rating_authority_error: str | None = None
    try:
        semantic_rating_authority = load_active_semantic_rating_authority_v1(
            root=root,
            environment=environment,
            as_of=as_of,
        )
    except (SemanticRatingAuthorityError, OSError, ValueError) as exc:
        semantic_rating_authority_error = str(exc)
    semantic_rating_active = (
        semantic_rating_authority.get("private_player_rating_authorized") is True
        and semantic_rating_authority.get("private_team_rating_authorized") is True
        and semantic_rating_authority.get("match_probability_authorized") is False
        and semantic_rating_authority.get("betting_authorized") is False
    )
    output_contract_trust: dict[str, Any] = {}
    output_contract_trust_error: str | None = None
    try:
        output_contract_trust = validate_current_contract_validation_inputs(
            repository_root=root
        )
    except (OSError, KeyError, TypeError, ValueError, ValidationFailure) as exc:
        output_contract_trust_error = str(exc)
    contract_reconciliation_candidate_record = _read_json(
        root, CONTRACT_RECONCILIATION_CANDIDATE_LOCATOR
    )
    contract_reconciliation_candidate = (
        contract_reconciliation_candidate_record.get("payload") or {}
    )
    contract_reconciliation_candidate_error: str | None = None
    if contract_reconciliation_candidate:
        try:
            contract_reconciliation_candidate = (
                validate_contract_reconciliation_candidate_v1(
                    contract_reconciliation_candidate, root=root
                )
            )
        except (OSError, ValueError, ContractReconciliationError) as exc:
            contract_reconciliation_candidate_error = str(exc)
            contract_reconciliation_candidate = {}
    contract_reference_replay = (
        contract_reconciliation_candidate.get("reference_semantic_replay") or {}
    )
    contract_prior_tree_recovery: dict[str, Any] = {}
    contract_prior_tree_recovery_error: str | None = None
    try:
        contract_prior_tree_recovery = load_prior_tree_recovery_v1(root=root)
    except (OSError, ValueError, ContractPriorTreeRecoveryError) as exc:
        contract_prior_tree_recovery_error = str(exc)
    contract_reconciliation_review: dict[str, Any] = {}
    contract_reconciliation_review_error: str | None = None
    try:
        contract_reconciliation_review = (
            load_pinned_contract_reconciliation_review_v1(
                root=root, environment=environment
            )
        )
    except (
        OSError,
        ValueError,
        ContractReconciliationReviewError,
    ) as exc:
        contract_reconciliation_review_error = str(exc)
    prospective_source_replay = (
        bool(v3_source_snapshot)
        and v3_preflight_v3.get("result_state")
        == "CORRECTED_SOURCE_PREFLIGHT_PASSED_NON_AUTHORIZING"
        and bool(v3_future_protocol)
        and (v3_future_protocol.get("supersession") or {}).get(
            "candidate_changed"
        )
        is False
        and (v3_future_protocol.get("supersession") or {}).get(
            "future_boundary_changed"
        )
        is False
        and v3_capture_implementation.get("ready_for_pre_event_capture") is True
    )
    source_pins_match = False
    source_pin_error: str | None = None
    artifact_input = multileague.get("input") or {}
    if multileague:
        try:
            if (
                artifact_input.get("maps_locator") != DEFAULT_MAPS_LOCATOR
                or artifact_input.get("players_locator") != DEFAULT_PLAYERS_LOCATOR
            ):
                raise ValueError("warehouse locators changed")
            maps_path = root / DEFAULT_MAPS_LOCATOR
            players_path = root / DEFAULT_PLAYERS_LOCATOR
            source_pins_match = (
                hashlib.sha256(maps_path.read_bytes()).hexdigest()
                == artifact_input.get("maps_sha256")
                and hashlib.sha256(players_path.read_bytes()).hexdigest()
                == artifact_input.get("players_sha256")
            )
        except (OSError, ValueError) as exc:
            source_pin_error = str(exc)

    component_semantics_valid = bool(teams)
    if isinstance(teams, list):
        for team in teams:
            components = (team or {}).get("components") or {}
            if (
                (components.get("player_aggregate") or {}).get("status") != "ESTIMATED"
                or (components.get("lineup_synergy") or {}).get("status") != "UNAVAILABLE"
                or (components.get("lineup_synergy") or {}).get("value") is not None
                or (components.get("team_policy") or {}).get("status") != "UNAVAILABLE"
                or (components.get("team_policy") or {}).get("value") is not None
            ):
                component_semantics_valid = False
                break
    else:
        component_semantics_valid = False

    checks = {
        "player_artifact_present_and_valid": bool(multileague),
        "warehouse_source_pins_match_current_files": source_pins_match,
        "prospective_source_snapshot_replaces_mutable_warehouse_dependency": (
            prospective_source_replay
        ),
        "semantic_output_contract_trust_root_current_and_valid": (
            bool(output_contract_trust) and output_contract_trust_error is None
        ),
        "semantic_output_contract_reconciliation_candidate_present_and_valid": (
            bool(contract_reconciliation_candidate)
            and contract_reconciliation_candidate_error is None
        ),
        "semantic_output_contract_candidate_reference_replay_passed": (
            contract_reference_replay.get("all_pass") is True
            and contract_reference_replay.get("generated_by_evaluated_system")
            is True
            and contract_reference_replay.get("independent_review_eligible")
            is False
            and contract_reference_replay.get("authority_granted") is False
        ),
        "semantic_output_contract_exact_prior_tree_recovered": (
            bool(contract_prior_tree_recovery)
            and contract_prior_tree_recovery_error is None
            and contract_prior_tree_recovery.get("contract_tree_sha256")
            == contract_reconciliation_candidate.get("active_trust_root", {}).get(
                "contract_tree_sha256"
            )
            and contract_prior_tree_recovery.get("runner_provenance", {}).get(
                "generated_by_evaluated_system"
            )
            is True
            and contract_prior_tree_recovery.get("runner_provenance", {}).get(
                "independent_review_eligible"
            )
            is False
        ),
        "semantic_output_contract_reconciliation_independently_reviewed": (
            bool(contract_reconciliation_review)
            and contract_reconciliation_review_error is None
            and contract_reconciliation_review.get("contract_trust_root_active")
            is False
        ),
        "semantic_rating_deployment_authority_active": semantic_rating_active,
        "player_rating_not_development_only": semantic_rating_active,
        "player_final_holdout_available": bool(v3_joint_evaluation_result),
        "player_reliability_evidence_present": bool(
            joint_ratings.get("reliability") or validation_comparison
        ),
        "overall_validation_uncertainty_gate_passed": (
            joint_rating_stratum_passed("overall")
            and joint_ratings.get("reliability_gate_passed") is True
        ),
        "lcs_validation_uncertainty_gate_passed": (
            joint_rating_stratum_passed("league:LCS")
        ),
        "strong_baseline_benchmark_present_and_valid": bool(benchmark),
        "v3_failed_v1_source_evidence_preserved": (
            bool(v3_source_snapshot_v1)
            and bool(v3_protocol_v1)
            and v3_preflight_v1.get("result_state")
            == "SOURCE_SCHEMA_PREFLIGHT_FAILED"
        ),
        "v3_replayable_joint_source_snapshot_present": bool(v3_source_snapshot),
        "v3_temporal_failure_receipt_present_and_valid": (
            v3_temporal_failure.get("result_state")
            == "FUTURE_DATED_RECEIPTS_REJECTED_AND_SUPERSESSION_REQUIRED"
            and (v3_temporal_failure.get("policy") or {}).get(
                "artifacts_qualify_as_future_evidence"
            )
            is False
        ),
        "v3_future_dated_v2_receipts_rejected": (
            set(v3_temporal_failures)
            == {
                "corrected_source_preflight_v2",
                "future_protocol_v2",
                "capture_readiness_v1",
            }
            and all(
                item.get("failure")
                == "artifact_existed_before_its_declared_lock_or_build_time"
                for item in v3_temporal_failures.values()
            )
        ),
        "v3_corrected_source_preflight_passed": (
            v3_preflight_v3.get("result_state")
            == "CORRECTED_SOURCE_PREFLIGHT_PASSED_NON_AUTHORIZING"
            and (v3_preflight_v3.get("future_boundary") or {}).get(
                "future_holdout_targets_accessed"
            )
            is False
        ),
        "v3_corrected_adaptive_candidate_diagnostic_present_and_valid": (
            v3_corrected_adaptive_diagnostic.get("result_state")
            == "INCUMBENT_RETAINED_NO_ADAPTIVE_SUPERSESSION_EVIDENCE"
        ),
        "v3_corrected_adaptive_diagnostic_preserves_future_holdout": (
            bool(v3_corrected_adaptive_diagnostic)
            and (
                v3_corrected_adaptive_diagnostic.get("information_boundary")
                or {}
            ).get("future_holdout_targets_accessed")
            is False
            and (
                v3_corrected_adaptive_diagnostic.get("retention_decision")
                or {}
            ).get("status")
            == "RETAIN_REGISTERED_INCUMBENT"
            and (
                v3_corrected_adaptive_diagnostic.get("retention_decision")
                or {}
            ).get("does_not_validate_incumbent")
            is True
        ),
        "v3_future_protocol_lock_present_and_valid": bool(v3_future_protocol),
        "v3_protocol_supersession_preserves_candidate_and_boundary": (
            bool(v3_future_protocol)
            and (v3_future_protocol.get("supersession") or {}).get(
                "candidate_changed"
            )
            is False
            and (v3_future_protocol.get("supersession") or {}).get(
                "future_boundary_changed"
            )
            is False
            and (v3_future_protocol.get("supersession") or {}).get(
                "future_outcomes_used_for_recovery"
            )
            is False
            and (v3_future_protocol.get("supersession") or {}).get(
                "rejected_artifacts_qualify_as_future_evidence"
            )
            is False
        ),
        "v3_future_targets_still_unopened": (
            bool(v3_future_protocol)
            and v3_future_holdout.get("status") == "EMPTY_NOT_YET_ACQUIRED"
            and bool(v3_decision_outputs)
            and all(value is None for value in v3_decision_outputs.values())
        ),
        "v3_future_holdout_support_met": (
            v3_live_ledger.get("status") == "SUPPORT_MET_OUTCOMES_UNOPENED"
        ),
        "v3_future_prediction_ledger_present_and_valid": bool(v3_live_ledger),
        "v3_joint_future_evaluation_independently_registered_and_passed": (
            joint_models_passed
        ),
        "v3_pre_event_prediction_ledger_capture_ready": (
            v3_capture_implementation.get("ready_for_pre_event_capture")
            is True
        ),
        "v3_prediction_and_ledger_system_clock_hardened": (
            v3_capture_contract.get(
                "prediction_system_clock_sampled_inside_builder"
            )
            is True
            and v3_capture_contract.get(
                "prediction_cli_user_timestamp_argument_present"
            )
            is False
            and v3_capture_contract.get(
                "ledger_system_clock_sampled_inside_builder"
            )
            is True
            and v3_capture_contract.get(
                "ledger_builder_user_timestamp_argument_present"
            )
            is False
        ),
        "v3_pre_event_prediction_ledger_has_eligible_entries": (
            bool(v3_live_ledger)
            and len(v3_live_ledger.get("entries") or []) > 0
        ),
        "v3_independent_protocol_review_present": (
            v3_joint_evaluation_registry.get(
                "phase_one_evaluation_independently_registered"
            )
            is True
        ),
        "v2_protocol_lock_present_and_valid": bool(protocol),
        "v2_protocol_artifact_integrity_valid": (
            bool(protocol) or protocol_integrity_without_source_replay
        ),
        "v2_protocol_source_replay_valid": bool(protocol),
        "v2_observed_validation_reclassified_as_adaptive_development": (
            protocol_disclosure.get(
                "presealed_outcomes_remain_adaptive_not_independent_validation"
            )
            is True
        ),
        "v2_sealed_final_targets_still_unopened": (
            protocol_boundary.get("sealed_final_targets_accessed") is False
            and protocol_final_gate.get("opened") is False
        ),
        "v2_adaptive_candidate_selection_artifact_present": (
            bool(selection_view)
        ),
        "v2_adaptive_selection_source_replay_valid": bool(v2_selection),
        "v2_adaptive_candidate_selected": (
            (selection_view.get("selection") or {}).get("selected_candidate_id")
            is not None
        ),
        "v2_independent_sealed_opening_approval_present": (
            sealed_opening_authority.get("status") == "registered"
            and sealed_opening_authority.get("sealed_evaluation_authorized") is True
        ),
        "v2_sealed_final_gate_passed": False,
        "player_beats_strong_organization_baseline_overall": (
            joint_rating_stratum_passed("overall")
        ),
        "player_beats_strong_organization_baseline_lcs": (
            joint_rating_stratum_passed("league:LCS")
        ),
        "player_beats_strong_organization_baseline_roster_change": (
            joint_rating_stratum_passed("roster_change")
        ),
        "team_artifact_present_and_valid": bool(multileague),
        "team_rating_available": semantic_rating_active,
        "team_last_observed_exact_roster_aggregation_available": bool(teams),
        "team_pre_event_exact_roster_aggregation_available": (
            semantic_rating_active
            and (root / REGISTRATIONS["rating_registry"][0]).is_file()
            and bool(environment.get(REGISTRATIONS["rating_registry"][1]))
            and (root / REGISTRATIONS["roster_registry"][0]).is_file()
            and bool(environment.get(REGISTRATIONS["roster_registry"][1]))
        ),
        "team_identified_estimand_preserves_unavailable_components": component_semantics_valid,
        "development_player_posterior_present": bool(players),
        "rating_registry_present": (
            root / REGISTRATIONS["rating_registry"][0]
        ).is_file(),
        "rating_registry_externally_pinned": bool(
            environment.get(REGISTRATIONS["rating_registry"][1])
        ),
        "roster_registry_present": (
            root / REGISTRATIONS["roster_registry"][0]
        ).is_file(),
        "roster_registry_externally_pinned": bool(
            environment.get(REGISTRATIONS["roster_registry"][1])
        ),
    }
    blockers = [
        name
        for name, passed in checks.items()
        if not passed
        and not (
            name == "warehouse_source_pins_match_current_files"
            and prospective_source_replay
        )
    ]
    return {
        "status": "ready_for_event_registration" if not blockers else "blocked",
        "rating_probability_authorized": False,
        "artifacts": {
            "player": {
                key: value
                for key, value in multileague_record.items()
                if key != "payload"
            }
            | {
                "schema_version": multileague.get("schema_version"),
                "declared_artifact_sha256": multileague.get("artifact_sha256"),
                "result_state": multileague.get("result_state"),
                "development_winner_candidate_id": winner_id,
                "validation_gate_passed": selection.get("validation_gate_passed"),
                "sealed_final_opened": selection.get("sealed_final_opened"),
                "source_pins_match": source_pins_match,
                "source_pin_error": source_pin_error,
                "legacy_mutable_source_comparison_is_promotion_gate": (
                    not prospective_source_replay
                ),
                "prospective_source_replay_available": prospective_source_replay,
            },
            "semantic_output_contract_trust_root": {
                "present_and_valid": bool(output_contract_trust),
                "error": output_contract_trust_error,
                "evidence": output_contract_trust,
                "model_or_betting_authority": False,
            },
            "semantic_output_contract_reconciliation_candidate": {
                "locator": CONTRACT_RECONCILIATION_CANDIDATE_LOCATOR.as_posix(),
                "present_and_valid": bool(contract_reconciliation_candidate),
                "raw_sha256": contract_reconciliation_candidate_record.get(
                    "raw_sha256"
                ),
                "artifact_sha256": contract_reconciliation_candidate.get(
                    "artifact_sha256"
                ),
                "result_state": contract_reconciliation_candidate.get(
                    "result_state"
                ),
                "decision": contract_reconciliation_candidate.get("decision"),
                "reference_semantic_replay": contract_reference_replay or None,
                "error": (
                    contract_reconciliation_candidate_error
                    or contract_reconciliation_candidate_record.get("error")
                ),
                "model_or_betting_authority": False,
            },
            "semantic_output_contract_reconciliation_review": {
                "locator": CONTRACT_RECONCILIATION_REVIEW_LOCATOR.as_posix(),
                "external_digest_environment": (
                    CONTRACT_RECONCILIATION_REVIEW_ENV
                ),
                "present_and_valid": bool(contract_reconciliation_review),
                "error": contract_reconciliation_review_error,
                "contract_trust_root_active": contract_reconciliation_review.get(
                    "contract_trust_root_active"
                ),
                "model_or_betting_authority": False,
            },
            "semantic_output_contract_prior_tree_recovery": {
                "locator": CONTRACT_PRIOR_TREE_RECOVERY_LOCATOR.as_posix(),
                "present_and_valid": bool(contract_prior_tree_recovery),
                "contract_tree_sha256": contract_prior_tree_recovery.get(
                    "contract_tree_sha256"
                ),
                "reconstruction": contract_prior_tree_recovery.get(
                    "reconstruction"
                ),
                "error": contract_prior_tree_recovery_error,
                "independent_review_authority": False,
                "model_or_betting_authority": False,
            },
            "team": {
                key: value
                for key, value in multileague_record.items()
                if key != "payload"
            }
            | {
                "schema_version": multileague.get("schema_version"),
                "declared_artifact_sha256": multileague.get("artifact_sha256"),
                "result_state": multileague.get("result_state"),
                "last_observed_exact_roster_teams": len(teams or []),
                "component_semantics_valid": component_semantics_valid,
            },
            "strong_baseline_benchmark": {
                key: value
                for key, value in benchmark_record.items()
                if key != "payload"
            }
            | {
                "schema_version": benchmark.get("schema_version"),
                "declared_artifact_sha256": benchmark.get("artifact_sha256"),
                "result_state": benchmark.get("result_state"),
                "selected_organization_candidate_id": (
                    benchmark.get("selection") or {}
                ).get("organization_candidate_id"),
                "validation_gate_passed": (
                    benchmark.get("selection") or {}
                ).get("validation_gate_passed"),
                "sealed_final_opened": (
                    benchmark.get("selection") or {}
                ).get("sealed_final_opened"),
            },
            "v2_protocol_lock": {
                key: item
                for key, item in protocol_record.items()
                if key != "payload"
            }
            | {
                "schema_version": protocol_view.get("schema_version"),
                "declared_artifact_sha256": protocol_view.get("artifact_sha256"),
                "result_state": protocol_view.get("result_state"),
                "candidate_count": len(
                    (protocol_view.get("candidate_family") or {}).get("candidates") or []
                ),
                "artifact_integrity_without_source_replay": (
                    protocol_integrity_without_source_replay
                ),
                "observed_validation_status": protocol_disclosure.get("status"),
                "sealed_final_series": protocol_boundary.get("sealed_final_series"),
                "sealed_final_opened": protocol_final_gate.get("opened"),
            },
            "v2_adaptive_selection": {
                key: item
                for key, item in v2_selection_record.items()
                if key != "payload"
            }
            | {
                "schema_version": selection_view.get("schema_version"),
                "declared_artifact_sha256": selection_view.get("artifact_sha256"),
                "result_state": selection_view.get("result_state"),
                "eligible_candidate_ids": (
                    (selection_view.get("selection") or {}).get(
                        "eligible_candidate_ids"
                    )
                ),
                "selected_candidate_id": (
                    (selection_view.get("selection") or {}).get(
                        "selected_candidate_id"
                    )
                ),
                "sealed_final_opened": (
                    (selection_view.get("sealed_final") or {}).get("opened")
                ),
                "artifact_integrity_without_source_replay": (
                    selection_integrity_without_source_replay
                ),
            },
            "v2_sealed_opening_authority": sealed_opening_authority,
            "v3_source_snapshot": {
                "locator": V3_SOURCE_MANIFEST_V2_LOCATOR.as_posix(),
                "present_and_valid": bool(v3_source_snapshot),
                "error": v3_source_snapshot_error,
                "package_id": v3_source_snapshot.get("package_id"),
                "files": v3_source_snapshot.get("files"),
                "authority": v3_source_snapshot.get("authority"),
            },
            "v3_source_snapshot_v1_failed_lineage": {
                "locator": V3_SOURCE_MANIFEST_V1_LOCATOR.as_posix(),
                "present_and_valid": bool(v3_source_snapshot_v1),
                "error": v3_source_snapshot_v1_error,
                "package_id": v3_source_snapshot_v1.get("package_id"),
                "files": v3_source_snapshot_v1.get("files"),
                "authority": v3_source_snapshot_v1.get("authority"),
            },
            "v3_source_preflight_v1": {
                "locator": V3_PREFLIGHT_V1_LOCATOR.as_posix(),
                "present_and_valid": bool(v3_preflight_v1),
                "error": v3_preflight_v1_error,
                "result_state": v3_preflight_v1.get("result_state"),
                "artifact_sha256": v3_preflight_v1.get("artifact_sha256"),
                "diagnostic": v3_preflight_v1.get("diagnostic"),
                "outcome_access": v3_preflight_v1.get("outcome_access"),
                "authority": v3_preflight_v1.get("authority"),
            },
            "v3_temporal_failure": {
                "locator": V3_TEMPORAL_FAILURE_LOCATOR.as_posix(),
                "present_and_valid": bool(v3_temporal_failure),
                "error": v3_temporal_failure_error,
                "result_state": v3_temporal_failure.get("result_state"),
                "artifact_sha256": v3_temporal_failure.get("artifact_sha256"),
                "failures": v3_temporal_failure.get("failures"),
                "policy": v3_temporal_failure.get("policy"),
                "outcome_access": v3_temporal_failure.get("outcome_access"),
                "authority": v3_temporal_failure.get("authority"),
            },
            "v3_source_preflight_v2_rejected": {
                "locator": V3_PREFLIGHT_V2_REJECTED_LOCATOR.as_posix(),
                "present_and_semantically_valid": bool(v3_preflight_v2_rejected),
                "error": v3_preflight_v2_rejected_error,
                "result_state": v3_preflight_v2_rejected.get("result_state"),
                "artifact_sha256": v3_preflight_v2_rejected.get(
                    "artifact_sha256"
                ),
                "temporal_failure": v3_temporal_failures.get(
                    "corrected_source_preflight_v2"
                ),
                "qualifies_as_future_evidence": False,
                "authority": v3_preflight_v2_rejected.get("authority"),
            },
            "v3_source_preflight_v3": {
                "locator": V3_PREFLIGHT_V3_LOCATOR.as_posix(),
                "present_and_valid": bool(v3_preflight_v3),
                "error": v3_preflight_v3_error,
                "result_state": v3_preflight_v3.get("result_state"),
                "built_at_utc": v3_preflight_v3.get("built_at_utc"),
                "artifact_sha256": v3_preflight_v3.get("artifact_sha256"),
                "adapter_preflight": v3_preflight_v3.get("adapter_preflight"),
                "numerical_preflight": v3_preflight_v3.get(
                    "numerical_preflight"
                ),
                "authority": v3_preflight_v3.get("authority"),
            },
            "v3_corrected_adaptive_candidate_diagnostic": {
                "locator": V3_CORRECTED_ADAPTIVE_DIAGNOSTIC_LOCATOR.as_posix(),
                "present_and_valid": bool(v3_corrected_adaptive_diagnostic),
                "error": v3_corrected_adaptive_diagnostic_error,
                "result_state": v3_corrected_adaptive_diagnostic.get(
                    "result_state"
                ),
                "built_at_utc": v3_corrected_adaptive_diagnostic.get(
                    "built_at_utc"
                ),
                "artifact_sha256": v3_corrected_adaptive_diagnostic.get(
                    "artifact_sha256"
                ),
                "incumbent_candidate_id": (
                    v3_corrected_adaptive_diagnostic.get("incumbent") or {}
                ).get("candidate_id"),
                "adaptive_challenger_id": (
                    v3_corrected_adaptive_diagnostic.get(
                        "adaptive_challenger"
                    )
                    or {}
                ).get("candidate_id"),
                "retention_decision": v3_corrected_adaptive_diagnostic.get(
                    "retention_decision"
                ),
                "incumbent_versus_comparators": (
                    v3_corrected_adaptive_diagnostic.get("incumbent") or {}
                ).get("versus_comparators"),
                "authority": v3_corrected_adaptive_diagnostic.get(
                    "authority"
                ),
            },
            "v3_future_protocol_v1_superseded": {
                "locator": V3_PROTOCOL_V1_LOCATOR.as_posix(),
                "present_and_valid": bool(v3_protocol_v1),
                "error": v3_protocol_v1_error,
                "result_state": v3_protocol_v1.get("result_state"),
                "artifact_sha256": v3_protocol_v1.get("artifact_sha256"),
                "operational_status": (
                    "SUPERSEDED_AFTER_FAILED_SOURCE_SCHEMA_PREFLIGHT"
                    if v3_protocol_v1 and v3_preflight_v1
                    else None
                ),
            },
            "v3_future_protocol_v2_rejected": {
                "locator": V3_PROTOCOL_V2_REJECTED_LOCATOR.as_posix(),
                "present_and_semantically_valid": bool(
                    v3_future_protocol_v2_rejected
                ),
                "error": v3_future_protocol_v2_rejected_error,
                "result_state": v3_future_protocol_v2_rejected.get(
                    "result_state"
                ),
                "artifact_sha256": v3_future_protocol_v2_rejected.get(
                    "artifact_sha256"
                ),
                "temporal_failure": v3_temporal_failures.get(
                    "future_protocol_v2"
                ),
                "qualifies_as_future_evidence": False,
                "authority": v3_future_protocol_v2_rejected.get("authority"),
            },
            "v3_future_protocol": {
                "locator": V3_PROTOCOL_V3_LOCATOR.as_posix(),
                "present_and_valid": bool(v3_future_protocol),
                "error": v3_future_protocol_error,
                "result_state": v3_future_protocol.get("result_state"),
                "artifact_sha256": v3_future_protocol.get("artifact_sha256"),
                "locked_at_utc": v3_future_protocol.get("locked_at_utc"),
                "supersession": v3_future_protocol.get("supersession"),
                "clock_corrected_source_preflight": v3_future_protocol.get(
                    "clock_corrected_source_preflight"
                ),
                "future_holdout": v3_future_holdout,
                "prediction_ledger": v3_prediction_ledger,
                "opening_authority": v3_opening_authority,
                "decision_outputs": v3_decision_outputs,
            },
            "v3_capture_readiness_v1_rejected": {
                "locator": V3_CAPTURE_READINESS_V1_REJECTED_LOCATOR.as_posix(),
                "present": v3_capture_readiness_v1_record.get("present"),
                "raw_sha256": v3_capture_readiness_v1_record.get("raw_sha256"),
                "semantic_replay_error": v3_capture_readiness_v1_rejected_error,
                "result_state": v3_capture_readiness_v1_rejected.get(
                    "result_state"
                ),
                "artifact_sha256": v3_capture_readiness_v1_rejected.get(
                    "artifact_sha256"
                ),
                "temporal_failure": v3_temporal_failures.get(
                    "capture_readiness_v1"
                ),
                "qualifies_as_future_evidence": False,
                "authority": v3_capture_readiness_v1_rejected.get("authority"),
            },
            "v3_capture_readiness_v2_superseded": {
                "locator": V3_CAPTURE_READINESS_V2_SUPERSEDED_LOCATOR.as_posix(),
                "present": v3_capture_readiness_v2_record.get("present"),
                "raw_sha256": v3_capture_readiness_v2_record.get("raw_sha256"),
                "semantic_replay_error": (
                    v3_capture_readiness_v2_superseded_error
                ),
                "result_state": v3_capture_readiness_v2_superseded.get(
                    "result_state"
                ),
                "artifact_sha256": v3_capture_readiness_v2_superseded.get(
                    "artifact_sha256"
                ),
                "superseded_by": (
                    v3_capture_readiness.get("supersession") or {}
                ),
                "qualifies_as_current_implementation_evidence": False,
                "authority": v3_capture_readiness_v2_superseded.get(
                    "authority"
                ),
            },
            "v3_capture_readiness": {
                "locator": V3_CAPTURE_READINESS_V3_LOCATOR.as_posix(),
                "present_and_valid": bool(v3_capture_readiness),
                "error": v3_capture_readiness_error,
                "result_state": v3_capture_readiness.get("result_state"),
                "locked_at_utc": v3_capture_readiness.get("locked_at_utc"),
                "clock_attestation": v3_capture_readiness.get(
                    "clock_attestation"
                ),
                "supersession": v3_capture_readiness.get("supersession"),
                "artifact_sha256": v3_capture_readiness.get("artifact_sha256"),
                "capture_contract": v3_capture_readiness.get("capture_contract"),
                "implementation": v3_capture_implementation,
                "ledger_state": v3_capture_ledger_state,
                "authority": v3_capture_readiness.get("authority"),
            },
            "v3_future_prediction_ledger": {
                "locator": V3_PREDICTION_LEDGER_LOCATOR.as_posix(),
                "present_and_valid": bool(v3_live_ledger),
                "error": v3_live_ledger_error or v3_live_ledger_record.get("error"),
                "raw_sha256": v3_live_ledger_record.get("raw_sha256"),
                "artifact_sha256": v3_live_ledger.get("artifact_sha256"),
                "status": v3_live_ledger.get("status"),
                "entries": len(v3_live_ledger.get("entries") or []),
                "metadata_support": v3_live_ledger.get("metadata_support"),
                "outcomes_present": v3_live_ledger.get("outcomes_present"),
                "outcomes_accessed": v3_live_ledger.get("outcomes_accessed"),
            },
            "v3_joint_phase_one_evaluation": {
                "registry_locator": PHASE_ONE_EVALUATION_REGISTRY_LOCATOR.as_posix(),
                "registry_present_and_valid": bool(
                    v3_joint_evaluation_registry
                ),
                "external_digest_pin_present": bool(v3_joint_registry_digest),
                "error": v3_joint_evaluation_error,
                "independently_registered": v3_joint_evaluation_registry.get(
                    "phase_one_evaluation_independently_registered"
                ),
                "models_independently_passed": v3_joint_evaluation_registry.get(
                    "phase_one_models_independently_passed"
                ),
                "result_state": v3_joint_evaluation_result.get("result_state"),
                "result_artifact_sha256": v3_joint_evaluation_result.get(
                    "artifact_sha256"
                ),
                "ratings_evaluation": joint_ratings or None,
                "draft_evaluation": v3_joint_evaluation_result.get(
                    "draft_evaluation"
                ),
                "production_rating_authorized": semantic_rating_active,
            },
            "semantic_rating_authority": {
                "locator": SEMANTIC_RATING_AUTHORITY_LOCATOR.as_posix(),
                "external_digest_pin_present": bool(
                    environment.get(SEMANTIC_RATING_AUTHORITY_ENV)
                ),
                "active": semantic_rating_active,
                "error": semantic_rating_authority_error,
                "authority_id": (
                    semantic_rating_authority.get("receipt") or {}
                ).get("authority_id"),
                "receipt_raw_sha256": semantic_rating_authority.get(
                    "receipt_raw_sha256"
                ),
                "private_player_rating_authorized": (
                    semantic_rating_authority.get(
                        "private_player_rating_authorized"
                    )
                ),
                "private_team_rating_authorized": (
                    semantic_rating_authority.get(
                        "private_team_rating_authorized"
                    )
                ),
                "match_probability_authorized": (
                    semantic_rating_authority.get(
                        "match_probability_authorized"
                    )
                ),
                "betting_authorized": semantic_rating_authority.get(
                    "betting_authorized"
                ),
            },
            "legacy_lpl": {
                "player": {
                    key: value
                    for key, value in legacy_player_record.items()
                    if key != "payload"
                }
                | {
                    "schema_version": legacy_player.get("schema_version"),
                    "result_state": legacy_player.get("result_state"),
                },
                "team": {
                    key: value
                    for key, value in legacy_team_record.items()
                    if key != "payload"
                }
                | {
                    "schema_version": legacy_team.get("schema_version"),
                    "result_state": legacy_team.get("result_state"),
                },
            },
        },
        "checks": checks,
        "blockers": blockers,
    }


def _live_totals_readiness(
    root: Path, as_of: datetime, environment: Mapping[str, str]
) -> dict[str, Any]:
    artifact_record = _read_json(root, LIVE_TOTALS_ARTIFACT)
    artifact = artifact_record.get("payload") or {}
    candidate_pin_valid = False
    candidate_pin_error: str | None = None
    if artifact:
        try:
            artifact = validate_development_candidate(root)
            candidate_pin_valid = True
        except (LiveTotalsCandidateError, OSError, ValueError) as exc:
            candidate_pin_error = str(exc)
            artifact_record["error"] = f"candidate_registry_invalid:{exc}"
            artifact_record["payload"] = None
            artifact = {}
    cutoffs = (artifact.get("meta") or {}).get("data_cutoff_by_league") or {}
    cutoff_ages: dict[str, float | None] = {}
    fresh_leagues: list[str] = []
    for league, value in cutoffs.items():
        try:
            cutoff = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            if cutoff.tzinfo is None:
                raise ValueError("timezone missing")
            age_days = (as_of - cutoff.astimezone(timezone.utc)).total_seconds() / 86400
        except ValueError:
            cutoff_ages[str(league)] = None
            continue
        cutoff_ages[str(league)] = round(age_days, 3)
        if 0 <= age_days <= FRESHNESS_LIMIT_DAYS:
            fresh_leagues.append(str(league))
    clusters = artifact.get("calibration_residual_clusters")
    lcs_patch_counts: dict[str, dict[str, int]] = {}
    for minute, by_league in (artifact.get("test_patch_counts") or {}).items():
        if not isinstance(by_league, Mapping):
            continue
        counts = by_league.get("LCS")
        if not isinstance(counts, Mapping):
            continue
        lcs_patch_counts[str(minute)] = {
            str(patch): int(count)
            for patch, count in counts.items()
            if isinstance(count, int) and count >= 0
        }

    def patch_key(value: str) -> tuple[int, ...]:
        try:
            return tuple(int(part) for part in value.split("."))
        except ValueError:
            return (-1,)

    observed_patches = {
        patch for counts in lcs_patch_counts.values() for patch in counts
    }
    latest_lcs_patch = max(observed_patches, key=patch_key) if observed_patches else None
    latest_patch_counts = (
        {
            minute: counts.get(latest_lcs_patch, 0)
            for minute, counts in sorted(lcs_patch_counts.items())
        }
        if latest_lcs_patch is not None
        else {}
    )
    latest_patch_min_test_n = (
        min(latest_patch_counts.values()) if latest_patch_counts else 0
    )
    source = (artifact.get("meta") or {}).get("source") or {}
    checks = {
        "artifact_present_and_valid": artifact_record.get("payload") is not None,
        "development_candidate_code_pin_valid": candidate_pin_valid,
        "replayable_source_snapshot_present": (
            candidate_pin_valid and isinstance(source.get("snapshot_manifest"), Mapping)
        ),
        "series_cluster_schema_current": (
            artifact.get("schema_version") == "scryglass.live-total-kills.v2"
        ),
        "series_cluster_residuals_present": (
            isinstance(clusters, Mapping) and bool(clusters)
        ),
        "exact_checkpoint_policy_present": (
            (artifact.get("authority") or {}).get(
                "validated_minutes_are_exact_checkpoints_only"
            )
            is True
        ),
        "at_least_one_fresh_league": bool(fresh_leagues),
        "latest_lcs_patch_holdout_sufficient": (
            latest_patch_min_test_n >= MIN_PATCH_TEST_GAMES
        ),
        "model_independent_validation_authority_present": (
            ((artifact.get("authority") or {}).get("dependence_interval") or {}).get(
                "status"
            )
            == "independently_validated"
        ),
        "total_kills_market_authority_present": (
            root / REGISTRATIONS["total_kills_market_authority"][0]
        ).is_file(),
        "total_kills_market_authority_externally_pinned": bool(
            environment.get(REGISTRATIONS["total_kills_market_authority"][1])
        ),
        "quote_registry_present": (
            root / REGISTRATIONS["quote_registry"][0]
        ).is_file(),
        "quote_registry_externally_pinned": bool(
            environment.get(REGISTRATIONS["quote_registry"][1])
        ),
    }
    blockers = [name for name, passed in checks.items() if not passed]
    return {
        "status": "ready_for_event_replay" if not blockers else "blocked",
        "probability_authorized": False,
        "artifact": {
            key: value for key, value in artifact_record.items() if key != "payload"
        }
        | {
            "schema_version": artifact.get("schema_version"),
            "data_cutoff_by_league": cutoffs,
            "cutoff_age_days_by_league": cutoff_ages,
            "freshness_limit_days": FRESHNESS_LIMIT_DAYS,
            "fresh_leagues": sorted(fresh_leagues),
            "candidate_pin_error": candidate_pin_error,
            "source_snapshot": source,
            "latest_lcs_patch": latest_lcs_patch,
            "latest_lcs_patch_test_n_by_minute": latest_patch_counts,
            "latest_lcs_patch_min_test_n": latest_patch_min_test_n,
            "minimum_patch_test_games": MIN_PATCH_TEST_GAMES,
        },
        "checks": checks,
        "blockers": blockers,
    }


def _phase_one_collection_inventory(
    root: Path,
    environment: Mapping[str, str],
    as_of: datetime,
) -> dict[str, Any]:
    def count(prefix: object) -> int:
        directory = root / Path(str(prefix))
        if not directory.is_dir() or directory.is_symlink():
            return 0
        return sum(
            path.is_file() and not path.is_symlink()
            for path in directory.rglob("*.json")
        )

    operator_path = root / PROSPECTIVE_CAPTURE_SOURCE_LOCATOR
    attempt_directory = root / Path(str(PROSPECTIVE_CAPTURE_ATTEMPT_PREFIX))
    attempt_counts = {
        "valid_success": 0,
        "valid_failure": 0,
        "invalid": 0,
    }
    attempt_errors: list[dict[str, str]] = []
    if attempt_directory.is_dir() and not attempt_directory.is_symlink():
        for path in sorted(attempt_directory.rglob("*.json")):
            locator = path.relative_to(root).as_posix()
            try:
                if path.is_symlink() or not path.is_file():
                    raise ProspectiveCaptureError(
                        "capture attempt must be a regular non-symlink file"
                    )
                payload = json.loads(path.read_text(encoding="ascii"))
                checked = validate_prospective_capture_attempt(payload, root=root)
                if checked["status"] == "FAILED_CLOSED":
                    attempt_counts["valid_failure"] += 1
                else:
                    attempt_counts["valid_success"] += 1
            except (
                json.JSONDecodeError,
                ProspectiveCaptureError,
                OSError,
                ValueError,
            ) as exc:
                attempt_counts["invalid"] += 1
                attempt_errors.append({"locator": locator, "error": str(exc)})

    side_neutral_counts = {
        "valid_pre_side_envelopes": 0,
        "valid_side_bindings": 0,
        "valid_side_neutral_terminal_drafts": 0,
        "valid_complete_bundles": 0,
        "invalid_artifacts": 0,
    }
    side_neutral_errors: list[dict[str, str]] = []
    for prefix, validator, error_type, count_key in (
        (
            PRE_SIDE_ENVELOPE_PREFIX,
            validate_pre_side_rating_envelope,
            PreSideRatingEnvelopeError,
            "valid_pre_side_envelopes",
        ),
        (
            PRE_SIDE_BINDING_PREFIX,
            validate_pre_side_rating_binding,
            PreSideRatingBindingError,
            "valid_side_bindings",
        ),
        (
            SIDE_NEUTRAL_DRAFT_PREFIX,
            validate_side_neutral_draft_prediction,
            SideNeutralDraftPredictionError,
            "valid_side_neutral_terminal_drafts",
        ),
        (
            SIDE_NEUTRAL_BUNDLE_PREFIX,
            validate_side_neutral_capture_bundle,
            SideNeutralCaptureBundleError,
            "valid_complete_bundles",
        ),
    ):
        directory = root / Path(str(prefix))
        if not directory.is_dir() or directory.is_symlink():
            continue
        for path in sorted(directory.rglob("*.json")):
            locator = path.relative_to(root).as_posix()
            try:
                if path.is_symlink() or not path.is_file():
                    raise ValueError("side-neutral artifact must be a regular file")
                payload = json.loads(path.read_text(encoding="ascii"))
                validator(payload, root=root)
                side_neutral_counts[count_key] += 1
            except (
                json.JSONDecodeError,
                error_type,
                OSError,
                ValueError,
            ) as exc:
                side_neutral_counts["invalid_artifacts"] += 1
                side_neutral_errors.append(
                    {"locator": locator, "error": str(exc)}
                )

    envelope_operator_path = root / PRE_SIDE_ENVELOPE_SOURCE_LOCATOR
    binding_operator_path = root / PRE_SIDE_BINDING_SOURCE_LOCATOR
    draft_operator_path = root / SIDE_NEUTRAL_DRAFT_SOURCE_LOCATOR
    bundle_operator_path = root / SIDE_NEUTRAL_BUNDLE_SOURCE_LOCATOR
    reviewed_operator_path = root / SIDE_NEUTRAL_OPERATOR_SOURCE_LOCATOR
    side_neutral_protocol: dict[str, Any] = {}
    side_neutral_protocol_error: str | None = None
    try:
        side_neutral_protocol = validate_registered_side_neutral_protocol_v2(root=root)
    except (SideNeutralProtocolRegistryV2Error, OSError, ValueError) as exc:
        side_neutral_protocol_error = str(exc)
    side_neutral_review: dict[str, Any] = {}
    side_neutral_review_error: str | None = None
    try:
        side_neutral_review = load_active_side_neutral_protocol_review(
            root=root,
            environment=environment,
            as_of=as_of,
        )
    except (SideNeutralProtocolReviewError, OSError, ValueError) as exc:
        side_neutral_review_error = str(exc)
    side_neutral_admission_implementation: dict[str, Any] = {}
    side_neutral_admission_implementation_error: str | None = None
    try:
        side_neutral_admission_implementation = (
            validate_registered_side_neutral_collection_implementation(root=root)
        )
    except (
        SideNeutralCollectionImplementationRegistryError,
        OSError,
        ValueError,
    ) as exc:
        side_neutral_admission_implementation_error = str(exc)
    side_neutral_review_packet: dict[str, Any] = {}
    side_neutral_review_packet_error: str | None = None
    side_neutral_review_packet_path = root / SIDE_NEUTRAL_REVIEW_PACKET_LOCATOR
    if (
        side_neutral_review_packet_path.is_file()
        and not side_neutral_review_packet_path.is_symlink()
    ):
        try:
            side_neutral_review_packet = validate_side_neutral_review_packet(
                json.loads(side_neutral_review_packet_path.read_text(encoding="ascii")),
                root=root,
            )
        except (
            json.JSONDecodeError,
            SideNeutralReviewPacketError,
            OSError,
            ValueError,
        ) as exc:
            side_neutral_review_packet_error = str(exc)
    side_neutral_ledger: dict[str, Any] = {}
    side_neutral_ledger_error: str | None = None
    side_neutral_ledger_path = root / SIDE_NEUTRAL_LEDGER_LOCATOR
    if side_neutral_ledger_path.is_file() and not side_neutral_ledger_path.is_symlink():
        try:
            side_neutral_ledger = validate_side_neutral_ledger(
                json.loads(side_neutral_ledger_path.read_text(encoding="ascii")),
                root=root,
                environment=environment,
                as_of=as_of,
            )
        except (
            json.JSONDecodeError,
            SideNeutralLedgerError,
            SideNeutralProtocolReviewError,
            OSError,
            ValueError,
        ) as exc:
            side_neutral_ledger_error = str(exc)

    return {
        "contract": {
            "plan_schema_version": PHASE_ONE_PLAN_SCHEMA_VERSION,
            "event_bundle_schema_version": PHASE_ONE_BUNDLE_SCHEMA_VERSION,
            "joint_ledger_schema_version": PHASE_ONE_SNAPSHOT_SCHEMA_VERSION,
            "system_clocked_builders": True,
            "outcome_fields_recursively_rejected": True,
            "exact_ratings_bytes_must_be_embedded_by_terminal_draft": True,
            "registered_child_ledgers_rebuilt_in_one_snapshot": True,
            "self_authorizing": False,
        },
        "inventory": {
            "plan_prefix": PHASE_ONE_PLAN_PREFIX.as_posix(),
            "event_bundle_prefix": PHASE_ONE_BUNDLE_PREFIX.as_posix(),
            "joint_ledger_prefix": PHASE_ONE_SNAPSHOT_PREFIX.as_posix(),
            "unvalidated_plan_files": count(PHASE_ONE_PLAN_PREFIX),
            "unvalidated_event_bundle_files": count(PHASE_ONE_BUNDLE_PREFIX),
            "unvalidated_joint_ledger_files": count(PHASE_ONE_SNAPSHOT_PREFIX),
            "note": (
                "File counts are operational inventory only. They do not establish "
                "semantic validity, independent pinning, support, or opening authority."
            ),
        },
        "prospective_operator": {
            "source_locator": PROSPECTIVE_CAPTURE_SOURCE_LOCATOR,
            "source_present": operator_path.is_file() and not operator_path.is_symlink(),
            "source_raw_sha256": (
                _sha256_path(operator_path)
                if operator_path.is_file() and not operator_path.is_symlink()
                else None
            ),
            "stages": ["prepare", "draft", "map-start"],
            "schedule_order_is_not_side_authority": True,
            "exact_blue_red_rosters_required": True,
            "failure_attempts_are_ineligible_evidence": True,
            "atomic_no_clobber_outputs": True,
            "versioned_joint_and_child_ledger_candidates": True,
            "attempt_prefix": PROSPECTIVE_CAPTURE_ATTEMPT_PREFIX.as_posix(),
            "attempt_counts": attempt_counts,
            "attempt_errors": attempt_errors,
            "operator_self_authorizing": False,
        },
        "side_neutral_revision": {
            "pre_side_envelope_source_locator": PRE_SIDE_ENVELOPE_SOURCE_LOCATOR,
            "pre_side_envelope_source_present": (
                envelope_operator_path.is_file()
                and not envelope_operator_path.is_symlink()
            ),
            "pre_side_envelope_source_raw_sha256": (
                _sha256_path(envelope_operator_path)
                if envelope_operator_path.is_file()
                and not envelope_operator_path.is_symlink()
                else None
            ),
            "side_binding_source_locator": PRE_SIDE_BINDING_SOURCE_LOCATOR,
            "side_binding_source_present": (
                binding_operator_path.is_file()
                and not binding_operator_path.is_symlink()
            ),
            "side_binding_source_raw_sha256": (
                _sha256_path(binding_operator_path)
                if binding_operator_path.is_file()
                and not binding_operator_path.is_symlink()
                else None
            ),
            "terminal_draft_adapter_source_locator": (
                SIDE_NEUTRAL_DRAFT_SOURCE_LOCATOR
            ),
            "terminal_draft_adapter_source_present": (
                draft_operator_path.is_file() and not draft_operator_path.is_symlink()
            ),
            "terminal_draft_adapter_source_raw_sha256": (
                _sha256_path(draft_operator_path)
                if draft_operator_path.is_file()
                and not draft_operator_path.is_symlink()
                else None
            ),
            "complete_bundle_source_locator": SIDE_NEUTRAL_BUNDLE_SOURCE_LOCATOR,
            "complete_bundle_source_present": (
                bundle_operator_path.is_file()
                and not bundle_operator_path.is_symlink()
            ),
            "complete_bundle_source_raw_sha256": (
                _sha256_path(bundle_operator_path)
                if bundle_operator_path.is_file()
                and not bundle_operator_path.is_symlink()
                else None
            ),
            "review_gated_operator_source_locator": (
                SIDE_NEUTRAL_OPERATOR_SOURCE_LOCATOR
            ),
            "review_gated_operator_source_present": (
                reviewed_operator_path.is_file()
                and not reviewed_operator_path.is_symlink()
            ),
            "review_gated_operator_source_raw_sha256": (
                _sha256_path(reviewed_operator_path)
                if reviewed_operator_path.is_file()
                and not reviewed_operator_path.is_symlink()
                else None
            ),
            "review_gated_operator_stages": list(SIDE_NEUTRAL_OPERATOR_STAGES),
            "review_gated_operator_phase_one_bridge_stages": list(
                SIDE_NEUTRAL_PHASE_ONE_BRIDGE_STAGES
            ),
            "operator_requires_external_review_before_every_stage": True,
            "operator_rejects_pre_review_pre_side_capture": True,
            "operator_outputs_are_canonical_and_no_clobber": True,
            "pre_side_envelope_prefix": PRE_SIDE_ENVELOPE_PREFIX.as_posix(),
            "side_binding_prefix": PRE_SIDE_BINDING_PREFIX.as_posix(),
            "terminal_draft_prefix": SIDE_NEUTRAL_DRAFT_PREFIX.as_posix(),
            "complete_bundle_prefix": SIDE_NEUTRAL_BUNDLE_PREFIX.as_posix(),
            "both_orientations_sealed_at_one_system_clock_sample": True,
            "schedule_order_is_never_side_authority": True,
            "side_binding_selects_existing_child_without_refit": True,
            "binding_alone_is_ineligible_evaluation_evidence": True,
            "terminal_draft_and_actual_map_start_still_required": True,
            "atomic_no_clobber_outputs": True,
            "artifact_counts": side_neutral_counts,
            "artifact_errors": side_neutral_errors,
            "protocol_supersession_candidate_locator": (
                SIDE_NEUTRAL_PROTOCOL_LOCATOR.as_posix()
            ),
            "protocol_supersession_candidate_present": bool(side_neutral_protocol),
            "protocol_supersession_candidate_error": side_neutral_protocol_error,
            "protocol_supersession_candidate_raw_sha256": (
                SIDE_NEUTRAL_PROTOCOL_RAW_SHA256
                if side_neutral_protocol
                else None
            ),
            "protocol_supersession_candidate_artifact_sha256": (
                SIDE_NEUTRAL_PROTOCOL_ARTIFACT_SHA256
                if side_neutral_protocol
                else None
            ),
            "repository_code_pin_present": bool(side_neutral_protocol),
            "admission_implementation_code_pin_valid": bool(
                side_neutral_admission_implementation
            ),
            "admission_implementation_code_pin_error": (
                side_neutral_admission_implementation_error
            ),
            "admission_implementation_records": (
                side_neutral_admission_implementation.get("records")
            ),
            "independent_review_packet_locator": (
                SIDE_NEUTRAL_REVIEW_PACKET_LOCATOR.as_posix()
            ),
            "independent_review_packet_present_and_valid": bool(
                side_neutral_review_packet
            ),
            "independent_review_packet_error": side_neutral_review_packet_error,
            "independent_review_packet_artifact_sha256": (
                side_neutral_review_packet.get("artifact_sha256")
            ),
            "independent_review_locator": SIDE_NEUTRAL_REVIEW_LOCATOR.as_posix(),
            "independent_review_present_and_valid": bool(side_neutral_review),
            "independent_review_error": side_neutral_review_error,
            "independent_review_id": side_neutral_review.get("review_id"),
            "independent_reviewed_at_utc": side_neutral_review.get(
                "reviewed_at_utc"
            ),
            "independently_registered": bool(side_neutral_review),
            "reviewed_ledger_locator": SIDE_NEUTRAL_LEDGER_LOCATOR.as_posix(),
            "reviewed_ledger_present_and_valid": bool(side_neutral_ledger),
            "reviewed_ledger_error": side_neutral_ledger_error,
            "reviewed_ledger_eligible_map_count": (
                (side_neutral_ledger.get("qualification") or {}).get(
                    "eligible_map_count"
                )
                if side_neutral_ledger
                else 0
            ),
            "reviewed_ledger_support": side_neutral_ledger.get("support"),
            "self_authorizing": False,
        },
        "outcomes_accessed": False,
        "opening_authority": False,
        "betting_authority": False,
    }


def _match_winner_market_readiness(
    root: Path,
    environment: Mapping[str, str],
    as_of: datetime,
) -> dict[str, Any]:
    protocol: dict[str, Any] = {}
    protocol_error: str | None = None
    try:
        protocol = validate_registered_match_winner_future_protocol_v1(root=root)
    except (MatchWinnerFutureProtocolRegistryError, OSError, ValueError) as exc:
        protocol_error = str(exc)
    public_terms_snapshot: dict[str, Any] = {}
    public_terms_snapshot_error: str | None = None
    try:
        public_terms_snapshot = validate_registered_betano_terms_snapshot_v1(
            root=root
        )
    except (BetanoTermsSnapshotRegistryError, OSError, ValueError) as exc:
        public_terms_snapshot_error = str(exc)
    terms_authority: dict[str, Any] = {}
    terms_authority_error: str | None = None
    terms_authority_locator, terms_authority_env = REGISTRATIONS[
        "match_winner_bookmaker_terms"
    ]
    terms_authority_digest = environment.get(terms_authority_env)
    if (root / terms_authority_locator).is_file() and terms_authority_digest:
        try:
            terms_authority = load_pinned_betano_terms_authority_v1(
                path=root / terms_authority_locator,
                external_sha256=terms_authority_digest,
                root=root,
            )
        except (BetanoTermsAuthorityError, OSError, ValueError) as exc:
            terms_authority_error = str(exc)
    quote_adapter_candidate: dict[str, Any] = {}
    quote_adapter_candidate_error: str | None = None
    quote_adapter_candidate_record = _read_json(
        root, BETANO_QUOTE_ADAPTER_CANDIDATE_LOCATOR
    )
    try:
        quote_adapter_candidate = (
            validate_registered_betano_quote_adapter_candidate_v1(root=root)
        )
    except (
        BetanoQuoteAdapterCandidateRegistryError,
        OSError,
        ValueError,
    ) as exc:
        quote_adapter_candidate_error = str(exc)
    quote_adapter_registry: dict[str, Any] = {}
    quote_adapter_registry_error: str | None = None
    try:
        quote_adapter_registry = load_registered_betano_quote_adapter_v1(
            expected_registry_sha256=environment.get(
                "SCRYGLASS_PRIVATE_MATCH_WINNER_QUOTE_ADAPTER_SHA256"
            ),
            root=root,
        )
    except (BetanoQuoteAdapterRegistryError, OSError, ValueError) as exc:
        quote_adapter_registry_error = str(exc)
    collection_readiness: dict[str, Any] = {}
    collection_readiness_error: str | None = None
    try:
        collection_readiness = (
            validate_registered_phase_one_collection_readiness_v1(root=root)
        )
    except (
        PhaseOneCollectionReadinessRegistryError,
        OSError,
        ValueError,
    ) as exc:
        collection_readiness_error = str(exc)
    evaluation_readiness: dict[str, Any] = {}
    evaluation_readiness_error: str | None = None
    try:
        evaluation_readiness = (
            validate_registered_phase_one_evaluation_readiness_v1(root=root)
        )
    except (
        RegisteredPhaseOneEvaluationReadinessError,
        OSError,
        ValueError,
    ) as exc:
        evaluation_readiness_error = str(exc)
    probability_pipeline_readiness: dict[str, Any] = {}
    probability_pipeline_readiness_error: str | None = None
    try:
        probability_pipeline_readiness = (
            validate_registered_probability_pipeline_readiness_v1(root=root)
        )
    except (
        RegisteredProbabilityPipelineReadinessError,
        OSError,
        ValueError,
    ) as exc:
        probability_pipeline_readiness_error = str(exc)
    evaluation_registry: dict[str, Any] = {}
    evaluation_registry_error: str | None = None
    evaluation_registry_path = root / PHASE_ONE_EVALUATION_REGISTRY_LOCATOR
    evaluation_registry_digest = environment.get(
        PHASE_ONE_EVALUATION_REGISTRY_ENV
    )
    if evaluation_registry_path.is_file() and evaluation_registry_digest:
        try:
            registry_payload = json.loads(evaluation_registry_path.read_text())
            result_locator = (registry_payload.get("result_binding") or {}).get(
                "result_locator"
            )
            if not isinstance(result_locator, str):
                raise PhaseOneEvaluationRegistryError(
                    "evaluation registry result locator is missing"
                )
            binding = expected_result_binding(
                result_locator=result_locator, root=root
            )
            evaluation_registry = load_pinned_evaluation_registry(
                path=evaluation_registry_path,
                external_sha256=evaluation_registry_digest,
                expected_binding=binding,
            )
        except (
            json.JSONDecodeError,
            PhaseOneEvaluationRegistryError,
            OSError,
            ValueError,
        ) as exc:
            evaluation_registry_error = str(exc)
    calibration_registry: dict[str, Any] = {}
    calibration_registry_error: str | None = None
    calibration_registry_locator, calibration_registry_env = REGISTRATIONS[
        "match_winner_calibration_uncertainty"
    ]
    calibration_registry_path = root / calibration_registry_locator
    calibration_registry_digest = environment.get(calibration_registry_env)
    if calibration_registry_path.is_file() and calibration_registry_digest:
        try:
            registry_payload = json.loads(calibration_registry_path.read_text())
            binding_payload = registry_payload.get("binding") or {}
            recalibration_locator = (
                binding_payload.get("recalibration") or {}
            ).get("locator")
            verification_locator = (
                binding_payload.get("uncertainty_verification") or {}
            ).get("locator")
            fast_verification_locator = (
                binding_payload.get("fast_uncertainty_verification") or {}
            ).get("locator")
            if not isinstance(recalibration_locator, str) or not isinstance(
                verification_locator, str
            ) or not isinstance(fast_verification_locator, str):
                raise CalibrationUncertaintyRegistryError(
                    "calibration/uncertainty registry binding is missing"
                )
            binding = expected_calibration_uncertainty_binding(
                recalibration_artifact_locator=recalibration_locator,
                verification_uncertainty_locator=verification_locator,
                verification_fast_uncertainty_locator=fast_verification_locator,
                root=root,
                environment=environment,
            )
            calibration_registry = (
                load_pinned_calibration_uncertainty_registry(
                    path=calibration_registry_path,
                    external_sha256=calibration_registry_digest,
                    expected_binding=binding,
                )
            )
        except (
            json.JSONDecodeError,
            CalibrationUncertaintyRegistryError,
            OSError,
            ValueError,
        ) as exc:
            calibration_registry_error = str(exc)
    probability_registry: dict[str, Any] = {}
    probability_registry_error: str | None = None
    probability_registry_locator, probability_registry_env = REGISTRATIONS[
        "match_winner_event_probability_registry"
    ]
    probability_registry_path = root / probability_registry_locator
    probability_registry_digest = environment.get(probability_registry_env)
    if probability_registry_path.is_file() and probability_registry_digest:
        try:
            registry_payload = json.loads(probability_registry_path.read_text())
            entries_payload = registry_payload.get("entries") or []
            receipt_locators = [
                item.get("receipt_locator")
                for item in entries_payload
                if isinstance(item, Mapping)
            ]
            if (
                not receipt_locators
                or any(not isinstance(item, str) for item in receipt_locators)
            ):
                raise EventProbabilityRegistryV2Error(
                    "probability registry receipt inventory is missing"
                )
            expected_probability_entries = expected_event_probability_entries(
                receipt_locators=receipt_locators,
                root=root,
                environment=environment,
            )
            probability_registry = load_pinned_event_probability_registry_v2(
                path=probability_registry_path,
                external_sha256=probability_registry_digest,
                expected=expected_probability_entries,
            )
        except (
            json.JSONDecodeError,
            EventProbabilityRegistryV2Error,
            OSError,
            ValueError,
        ) as exc:
            probability_registry_error = str(exc)
    quote_registry: dict[str, Any] = {}
    quote_registry_error: str | None = None
    quote_registry_locator, quote_registry_env = REGISTRATIONS[
        "match_winner_quote_registry"
    ]
    quote_registry_path = root / quote_registry_locator
    quote_registry_digest = environment.get(quote_registry_env)
    if quote_registry_path.is_file() and quote_registry_digest:
        try:
            registry_payload = json.loads(quote_registry_path.read_text())
            qualification_locators = [
                item.get("qualification_locator")
                for item in (registry_payload.get("entries") or [])
                if isinstance(item, Mapping)
            ]
            if not qualification_locators or any(
                not isinstance(item, str) for item in qualification_locators
            ):
                raise BetanoQuoteRegistryV2Error(
                    "quote registry qualification inventory is missing"
                )
            expected_quotes = expected_betano_quote_entries(
                qualification_locators=qualification_locators,
                root=root,
                environment=environment,
            )
            quote_registry = load_pinned_betano_quote_registry_v2(
                path=quote_registry_path,
                external_sha256=quote_registry_digest,
                expected=expected_quotes,
            )
        except (
            json.JSONDecodeError,
            BetanoQuoteRegistryV2Error,
            OSError,
            ValueError,
        ) as exc:
            quote_registry_error = str(exc)
    phase_two_opening: dict[str, Any] = {}
    phase_two_opening_error: str | None = None
    try:
        phase_two_opening = validate_active_phase_two_opening(
            root=root, environment=environment
        )
    except (PhaseTwoOpeningError, OSError, ValueError) as exc:
        phase_two_opening_error = str(exc)
    phase_two_collection_readiness: dict[str, Any] = {}
    phase_two_collection_readiness_error: str | None = None
    collection_readiness_path = root / PHASE_TWO_COLLECTION_READINESS_REGISTRY_LOCATOR
    collection_readiness_digest = environment.get(
        PHASE_TWO_COLLECTION_READINESS_ENV
    )
    if collection_readiness_path.is_file() and collection_readiness_digest:
        try:
            binding = expected_phase_two_collection_readiness_binding(
                root=root, environment=environment
            )
            phase_two_collection_readiness = (
                load_pinned_phase_two_collection_readiness_registry_v1(
                    path=collection_readiness_path,
                    external_sha256=collection_readiness_digest,
                    expected_binding=binding,
                )
            )
        except (
            PhaseTwoCollectionReadinessRegistryError,
            OSError,
            ValueError,
        ) as exc:
            phase_two_collection_readiness_error = str(exc)
    phase_two_evaluation_readiness: dict[str, Any] = {}
    phase_two_evaluation_readiness_error: str | None = None
    evaluation_readiness_path = root / PHASE_TWO_EVALUATION_READINESS_REGISTRY_LOCATOR
    evaluation_readiness_digest = environment.get(
        PHASE_TWO_EVALUATION_READINESS_ENV
    )
    if evaluation_readiness_path.is_file() and evaluation_readiness_digest:
        try:
            binding = expected_phase_two_evaluation_readiness_binding(
                root=root, environment=environment
            )
            phase_two_evaluation_readiness = (
                load_pinned_phase_two_evaluation_readiness_registry_v1(
                    path=evaluation_readiness_path,
                    external_sha256=evaluation_readiness_digest,
                    expected_binding=binding,
                )
            )
        except (
            PhaseTwoEvaluationReadinessRegistryError,
            OSError,
            ValueError,
        ) as exc:
            phase_two_evaluation_readiness_error = str(exc)
    phase_two_snapshot_registry: dict[str, Any] = {}
    phase_two_snapshot_registry_error: str | None = None
    snapshot_registry_path = root / PHASE_TWO_SNAPSHOT_REGISTRY_LOCATOR
    snapshot_registry_digest = environment.get(PHASE_TWO_SNAPSHOT_REGISTRY_ENV)
    if snapshot_registry_path.is_file() and snapshot_registry_digest:
        try:
            registry_payload = json.loads(snapshot_registry_path.read_text())
            snapshot_locator = (
                registry_payload.get("snapshot_binding") or {}
            ).get("snapshot_locator")
            if not isinstance(snapshot_locator, str):
                raise PhaseTwoSnapshotRegistryError(
                    "stopping-snapshot registry locator is missing"
                )
            binding = expected_phase_two_snapshot_binding(
                snapshot_locator=snapshot_locator,
                root=root,
                environment=environment,
            )
            phase_two_snapshot_registry = (
                load_pinned_phase_two_snapshot_registry_v1(
                    path=snapshot_registry_path,
                    external_sha256=snapshot_registry_digest,
                    expected_binding=binding,
                )
            )
        except (
            json.JSONDecodeError,
            PhaseTwoSnapshotRegistryError,
            OSError,
            ValueError,
        ) as exc:
            phase_two_snapshot_registry_error = str(exc)
    phase_two_evaluation_registry: dict[str, Any] = {}
    phase_two_evaluation_registry_error: str | None = None
    phase_two_evaluation_path = root / PHASE_TWO_EVALUATION_REGISTRY_LOCATOR
    phase_two_evaluation_digest = environment.get(PHASE_TWO_EVALUATION_REGISTRY_ENV)
    if phase_two_evaluation_path.is_file() and phase_two_evaluation_digest:
        try:
            registry_payload = json.loads(phase_two_evaluation_path.read_text())
            result_locator = (
                registry_payload.get("result_binding") or {}
            ).get("result_locator")
            if not isinstance(result_locator, str):
                raise PhaseTwoEvaluationRegistryError(
                    "phase-two evaluation result locator is missing"
                )
            binding = expected_phase_two_result_binding(
                result_locator=result_locator,
                root=root,
                environment=environment,
            )
            phase_two_evaluation_registry = (
                load_pinned_phase_two_evaluation_registry_v1(
                    path=phase_two_evaluation_path,
                    external_sha256=phase_two_evaluation_digest,
                    expected_binding=binding,
                )
            )
        except (
            json.JSONDecodeError,
            PhaseTwoEvaluationRegistryError,
            OSError,
            ValueError,
        ) as exc:
            phase_two_evaluation_registry_error = str(exc)
    semantic_market_authority: dict[str, Any] = {}
    semantic_market_authority_error: str | None = None
    try:
        semantic_market_authority = load_active_semantic_market_authority_v1(
            root=root,
            environment=environment,
            as_of=as_of,
        )
    except (SemanticMarketAuthorityError, OSError, ValueError) as exc:
        semantic_market_authority_error = str(exc)
    phase_one = protocol.get("phase_one") or {}
    phase_two = protocol.get("phase_two") or {}
    quote_contract = protocol.get("quote_capture_contract") or {}
    settlement_contract = protocol.get("settlement_contract") or {}
    registries = protocol.get("registries") or {}
    phase_one_collection = _phase_one_collection_inventory(
        root,
        environment,
        as_of,
    )
    prospective_operator = phase_one_collection["prospective_operator"]
    side_neutral_revision = phase_one_collection["side_neutral_revision"]

    def registration_ready(name: str) -> tuple[bool, bool]:
        locator, environment_name = REGISTRATIONS[name]
        return (root / locator).is_file(), bool(environment.get(environment_name))

    phase_one_present, phase_one_pinned = registration_ready(
        "match_winner_phase_one_evaluation"
    )
    calibration_present, calibration_pinned = registration_ready(
        "match_winner_calibration_uncertainty"
    )
    terms_present, terms_pinned = registration_ready(
        "match_winner_bookmaker_terms"
    )
    adapter_present, adapter_pinned = registration_ready(
        "match_winner_quote_adapter"
    )
    probability_present, probability_pinned = registration_ready(
        "match_winner_event_probability_registry"
    )
    phase_two_present, phase_two_pinned = registration_ready(
        "match_winner_phase_two_opening"
    )
    phase_two_collection_readiness_present, phase_two_collection_readiness_pinned = registration_ready(
        "match_winner_phase_two_collection_readiness"
    )
    phase_two_evaluation_readiness_present, phase_two_evaluation_readiness_pinned = registration_ready(
        "match_winner_phase_two_evaluation_readiness"
    )
    phase_two_snapshot_present, phase_two_snapshot_pinned = registration_ready(
        "match_winner_phase_two_stopping_snapshot"
    )
    phase_two_evaluation_present, phase_two_evaluation_pinned = registration_ready(
        "match_winner_phase_two_evaluation"
    )
    quote_present, quote_pinned = registration_ready(
        "match_winner_quote_registry"
    )
    authority_present, authority_pinned = registration_ready(
        "match_winner_market_authority"
    )

    checks = {
        "future_market_protocol_locked_and_valid": bool(protocol),
        "future_market_protocol_raw_hash_registered": (
            bool(protocol)
            and _sha256_path(root / MATCH_WINNER_PROTOCOL_LOCATOR)
            == MATCH_WINNER_PROTOCOL_RAW_SHA256
        ),
        "future_market_protocol_artifact_hash_registered": (
            protocol.get("artifact_sha256")
            == MATCH_WINNER_PROTOCOL_ARTIFACT_SHA256
        ),
        "public_scryglass_remains_non_betting": (
            (protocol.get("scope") or {}).get(
                "public_scryglass_remains_non_betting"
            )
            is True
        ),
        "protocol_itself_grants_no_authority": (
            bool(protocol)
            and all(value is False for value in (protocol.get("authority") or {}).values())
            and all(
                value is None
                for value in (protocol.get("decision_outputs") or {}).values()
            )
        ),
        "phase_one_future_outcomes_still_sealed": (
            phase_one.get("status") == "EMPTY_OUTCOMES_SEALED"
        ),
        "outcome_free_phase_one_collection_contract_present": (
            phase_one_collection["contract"] == {
                "plan_schema_version": (
                    "scryglass:match-winner-phase-one-event-plan:v1"
                ),
                "event_bundle_schema_version": (
                    "scryglass:match-winner-phase-one-event-bundle:v1"
                ),
                "joint_ledger_schema_version": (
                    "scryglass:match-winner-phase-one-joint-ledger:v1"
                ),
                "system_clocked_builders": True,
                "outcome_fields_recursively_rejected": True,
                "exact_ratings_bytes_must_be_embedded_by_terminal_draft": True,
                "registered_child_ledgers_rebuilt_in_one_snapshot": True,
                "self_authorizing": False,
            }
        ),
        "prospective_phase_one_capture_operator_present_and_fail_closed": (
            prospective_operator.get("source_present") is True
            and prospective_operator.get("schedule_order_is_not_side_authority")
            is True
            and prospective_operator.get("exact_blue_red_rosters_required") is True
            and prospective_operator.get("failure_attempts_are_ineligible_evidence")
            is True
            and prospective_operator.get("atomic_no_clobber_outputs") is True
            and prospective_operator.get("operator_self_authorizing") is False
            and (prospective_operator.get("attempt_counts") or {}).get("invalid")
            == 0
        ),
        "side_neutral_pre_side_and_binding_implementations_fail_closed": (
            side_neutral_revision.get("pre_side_envelope_source_present") is True
            and side_neutral_revision.get("side_binding_source_present") is True
            and side_neutral_revision.get("terminal_draft_adapter_source_present")
            is True
            and side_neutral_revision.get("complete_bundle_source_present") is True
            and side_neutral_revision.get(
                "both_orientations_sealed_at_one_system_clock_sample"
            )
            is True
            and side_neutral_revision.get(
                "schedule_order_is_never_side_authority"
            )
            is True
            and side_neutral_revision.get(
                "side_binding_selects_existing_child_without_refit"
            )
            is True
            and side_neutral_revision.get(
                "binding_alone_is_ineligible_evaluation_evidence"
            )
            is True
            and side_neutral_revision.get("self_authorizing") is False
            and (side_neutral_revision.get("artifact_counts") or {}).get(
                "invalid_artifacts"
            )
            == 0
        ),
        "side_neutral_review_gated_end_to_end_operator_present": (
            side_neutral_revision.get("review_gated_operator_source_present")
            is True
            and side_neutral_revision.get("review_gated_operator_stages")
            == ["pre-side", "bind-side", "draft", "map-start", "ledger"]
            and side_neutral_revision.get(
                "operator_requires_external_review_before_every_stage"
            )
            is True
            and side_neutral_revision.get(
                "operator_rejects_pre_review_pre_side_capture"
            )
            is True
            and side_neutral_revision.get(
                "operator_outputs_are_canonical_and_no_clobber"
            )
            is True
        ),
        "side_neutral_reviewed_operator_feeds_frozen_phase_one_evaluator": (
            side_neutral_revision.get(
                "review_gated_operator_phase_one_bridge_stages"
            )
            == ["draft", "map-start", "ledger"]
            and side_neutral_revision.get(
                "operator_outputs_are_canonical_and_no_clobber"
            )
            is True
            and side_neutral_revision.get("review_gated_operator_source_present")
            is True
        ),
        "side_neutral_protocol_supersession_independently_registered": (
            side_neutral_revision.get("protocol_supersession_candidate_present")
            is True
            and side_neutral_revision.get("independently_registered") is True
        ),
        "side_neutral_review_packet_and_admission_code_pins_valid": (
            side_neutral_revision.get("independent_review_packet_present_and_valid")
            is True
            and side_neutral_revision.get(
                "admission_implementation_code_pin_valid"
            )
            is True
        ),
        "side_neutral_reviewed_ledger_present_and_valid": (
            side_neutral_revision.get("reviewed_ledger_present_and_valid") is True
        ),
        "side_neutral_reviewed_ledger_has_eligible_entries": (
            (side_neutral_revision.get("reviewed_ledger_eligible_map_count") or 0)
            > 0
        ),
        "phase_one_collection_readiness_locked_empty_and_valid": (
            bool(collection_readiness)
            and collection_readiness.get("artifact_sha256")
            == PHASE_ONE_READINESS_ARTIFACT_SHA256
            and _sha256_path(root / PHASE_ONE_READINESS_LOCATOR)
            == PHASE_ONE_READINESS_RAW_SHA256
            and collection_readiness.get("locked_empty_collection_state")
            == {
                "plans": 0,
                "event_bundles": 0,
                "joint_snapshots": 0,
                "outcomes_present": False,
                "outcomes_accessed": False,
                "metadata_support_met": False,
                "independently_pinned": False,
                "opening_authority": False,
            }
            and collection_readiness.get("implementation", {}).get(
                "ready_for_outcome_free_phase_one_collection"
            )
            is True
            and all(
                value is False
                for value in (collection_readiness.get("authority") or {}).values()
            )
        ),
        "phase_one_evaluation_readiness_locked_empty_and_valid": (
            bool(evaluation_readiness)
            and evaluation_readiness.get("artifact_sha256")
            == PHASE_ONE_EVALUATION_READINESS_ARTIFACT_SHA256
            and _sha256_path(root / PHASE_ONE_EVALUATION_READINESS_LOCATOR)
            == PHASE_ONE_EVALUATION_READINESS_RAW_SHA256
            and evaluation_readiness.get("locked_empty_state")
            == {
                "parity_registries": 0,
                "outcome_cohorts": 0,
                "outcome_evidence": 0,
                "opening_markers": 0,
                "evaluation_outputs": 0,
                "opening_authority_present": False,
                "outcomes_accessed": False,
            }
            and all(
                value is False
                for value in (evaluation_readiness.get("authority") or {}).values()
            )
        ),
        "post_pass_probability_pipeline_implementation_frozen_pre_boundary": (
            bool(probability_pipeline_readiness)
            and probability_pipeline_readiness.get("artifact_sha256")
            == PROBABILITY_PIPELINE_READINESS_ARTIFACT_SHA256
            and _sha256_path(root / PROBABILITY_PIPELINE_READINESS_LOCATOR)
            == PROBABILITY_PIPELINE_READINESS_RAW_SHA256
            and probability_pipeline_readiness.get("locked_empty_state")
            == {
                "phase_one_outcome_cohorts": 0,
                "phase_one_evaluation_outputs": 0,
                "recalibration_artifacts": 0,
                "event_uncertainty_artifacts": 0,
                "phase_one_evaluation_registry_present": False,
                "recalibration_uncertainty_registry_present": False,
                "phase_two_opening_present": False,
                "phase_two_started": False,
                "outcomes_accessed": False,
            }
            and (
                probability_pipeline_readiness.get(
                    "probability_pipeline_contract", {}
                )
                .get("uncertainty", {})
                .get("resamples")
                == 2_000
            )
            and all(
                value is False
                for value in (
                    probability_pipeline_readiness.get("authority") or {}
                ).values()
            )
        ),
        "fresh_post_validation_rating_refit_requirement_frozen": (
            (
                probability_pipeline_readiness.get(
                    "probability_pipeline_contract", {}
                ).get("fresh_post_validation_rating_refit")
                or {}
            )
            == {
                "required_before_every_phase_two_event_prediction": True,
                "model_family_and_hyperparameters_fixed_by_phase_one": True,
                "independently_registered_phase_one_pass_required": True,
                "immutable_exact_source_bytes_required": True,
                "strict_target_event_cutoff": True,
                "availability_embargo_hours": 48,
                "maximum_data_age_seconds": 14 * 24 * 60 * 60,
                "exact_pre_event_roster_and_patch_receipts_required": True,
                "cross_team_covariance_retained": True,
                "unidentified_synergy_and_policy_remain_null": True,
                "match_probability_or_betting_authority": False,
                "full_pipeline_uncertainty_binding_status": (
                    "wired_replayed_and_independently_reviewable"
                ),
            }
        ),
        "fresh_post_validation_rating_refit_full_pipeline_binding_complete": (
            (
                probability_pipeline_readiness.get(
                    "probability_pipeline_contract", {}
                )
                .get("fresh_post_validation_rating_refit", {})
                .get("full_pipeline_uncertainty_binding_status")
                == "wired_replayed_and_independently_reviewable"
            )
        ),
        "phase_one_models_independently_passed": (
            phase_one_present
            and phase_one_pinned
            and evaluation_registry.get(
                "phase_one_evaluation_independently_registered"
            )
            is True
            and evaluation_registry.get("phase_one_models_independently_passed")
            is True
        ),
        "phase_two_not_started_before_phase_one": (
            phase_two.get("status") == "NOT_OPEN_NOT_STARTED"
            and registries.get("phase_two_prediction_ledger") is None
            and registries.get("phase_two_quote_ledger") is None
        ),
        "quote_capture_contract_hash_registered": (
            protocol.get("quote_capture_contract_sha256")
            == REGISTERED_QUOTE_CAPTURE_CONTRACT_SHA256
        ),
        "quote_builder_time_not_misrepresented_as_transport_time": (
            quote_contract.get(
                "generic_receipt_builder_time_counts_as_transport_time"
            )
            is False
            and quote_contract.get("user_supplied_capture_timestamp_allowed")
            is False
        ),
        "settlement_contract_hash_registered": (
            protocol.get("settlement_contract_sha256")
            == REGISTERED_SETTLEMENT_CONTRACT_SHA256
        ),
        "public_bookmaker_terms_snapshot_locked_and_valid": (
            bool(public_terms_snapshot)
            and public_terms_snapshot.get("artifact_sha256")
            == BETANO_TERMS_SNAPSHOT_ARTIFACT_SHA256
            and _sha256_path(root / BETANO_TERMS_SNAPSHOT_LOCATOR)
            == BETANO_TERMS_SNAPSHOT_RAW_SHA256
        ),
        "public_bookmaker_terms_snapshot_honestly_incomplete": (
            bool(public_terms_snapshot)
            and (public_terms_snapshot.get("coverage") or {}).get(
                "complete_bookmaker_terms_snapshot"
            )
            is False
            and (public_terms_snapshot.get("coverage") or {}).get(
                "independent_alignment_review_present"
            )
            is False
        ),
        "bookmaker_terms_snapshot_independently_registered": (
            terms_present
            and terms_pinned
            and terms_authority.get(
                "bookmaker_terms_snapshot_independently_registered"
            )
            is True
            and terms_authority.get("settlement_contract_resolved") is True
        ),
        "source_specific_quote_adapter_candidate_locked_and_valid": (
            bool(quote_adapter_candidate)
            and quote_adapter_candidate.get("artifact_sha256")
            == BETANO_QUOTE_ADAPTER_CANDIDATE_ARTIFACT_SHA256
            and quote_adapter_candidate_record.get("raw_sha256")
            == BETANO_QUOTE_ADAPTER_CANDIDATE_RAW_SHA256
            and quote_adapter_candidate.get("registration", {}).get(
                "independently_registered"
            )
            is False
            and quote_adapter_candidate.get("live_capture", {}).get(
                "first_phase_two_quote_created"
            )
            is False
            and all(
                value is False
                for value in (quote_adapter_candidate.get("authority") or {}).values()
            )
        ),
        "source_specific_quote_adapter_independently_registered": (
            adapter_present
            and adapter_pinned
            and bool(quote_adapter_registry)
            and quote_adapter_registry.get("authority", {}).get(
                "source_adapter_identity_authority"
            )
            is True
            and quote_adapter_registry.get("authority", {}).get(
                "betting_authority"
            )
            is False
        ),
        "phase_one_calibration_and_uncertainty_independently_registered": (
            calibration_present
            and calibration_pinned
            and calibration_registry.get(
                "recalibration_independently_registered"
            )
            is True
            and calibration_registry.get(
                "uncertainty_implementation_independently_registered"
            )
            is True
        ),
        "event_probability_receipt_and_registry_contract_present": (
            EVENT_PROBABILITY_RECEIPT_SCHEMA_VERSION
            == "scryglass:private-event-probability:v2"
            and EVENT_PROBABILITY_REGISTRY_SCHEMA_VERSION
            == "scryglass:private-event-probability-registry:v2"
            and MARKET_AUTHORITY_SCHEMA_VERSION
            == "scryglass.private-market-authority.v2"
            and BETANO_QUOTE_QUALIFICATION_SCHEMA_VERSION
            == "scryglass:betano-br-map-winner-quote-qualification:v1"
            and BETANO_QUOTE_REGISTRY_SCHEMA_VERSION
            == "scryglass:betano-br-map-winner-quote-registry:v2"
        ),
        "event_probability_registry_independently_registered": (
            probability_present
            and probability_pinned
            and probability_registry.get(
                "event_probability_identity_authority"
            )
            is True
        ),
        "quote_registry_independently_registered": (
            quote_present
            and quote_pinned
            and quote_registry.get("quote_identity_authority") is True
            and quote_registry.get("odds_accuracy_authorized") is False
            and quote_registry.get("betting_authorized") is False
        ),
        "phase_two_opening_independently_registered": (
            phase_two_present
            and phase_two_pinned
            and phase_two_opening.get(
                "outcome_free_phase_two_collection_active"
            )
            is True
            and phase_two_opening.get("probability_authorized") is False
            and phase_two_opening.get("betting_authorized") is False
        ),
        "phase_two_collection_readiness_independently_registered": (
            phase_two_collection_readiness_present
            and phase_two_collection_readiness_pinned
            and phase_two_collection_readiness.get(
                "phase_two_collection_readiness_independently_registered"
            )
            is True
            and phase_two_collection_readiness.get("betting_authorized") is False
        ),
        "phase_two_evaluation_readiness_independently_registered": (
            phase_two_evaluation_readiness_present
            and phase_two_evaluation_readiness_pinned
            and phase_two_evaluation_readiness.get(
                "phase_two_evaluation_readiness_independently_registered"
            )
            is True
            and phase_two_evaluation_readiness.get("betting_authorized") is False
        ),
        "phase_two_first_support_met_snapshot_independently_registered": (
            phase_two_snapshot_present
            and phase_two_snapshot_pinned
            and phase_two_snapshot_registry.get(
                "phase_two_snapshot_identity_authority"
            )
            is True
            and phase_two_snapshot_registry.get("betting_authorized") is False
        ),
        "phase_two_market_evaluation_independently_registered": (
            phase_two_evaluation_present
            and phase_two_evaluation_pinned
            and phase_two_evaluation_registry.get(
                "phase_two_evaluation_independently_registered"
            )
            is True
            and phase_two_evaluation_registry.get(
                "phase_two_market_gates_independently_passed"
            )
            is True
            and phase_two_evaluation_registry.get("betting_authorized") is False
        ),
        "match_winner_market_authority_independently_registered": (
            authority_present
            and authority_pinned
            and bool(semantic_market_authority)
            and semantic_market_authority.get(
                "private_probability_generation_authorized"
            )
            is True
            and semantic_market_authority.get(
                "private_decision_support_authorized"
            )
            is True
            and semantic_market_authority.get("transaction_authorized") is False
            and semantic_market_authority.get("stake_authorized") is False
        ),
    }
    blockers = [name for name, passed in checks.items() if not passed]
    return {
        "status": (
            "semantic_authority_active_event_replay_required"
            if semantic_market_authority
            else "protocol_locked_waiting_for_future_evaluation"
            if protocol
            else "blocked"
        ),
        "probability_authorized": False,
        "expected_value_authorized": False,
        "betting_authorized": False,
        "protocol": {
            "locator": MATCH_WINNER_PROTOCOL_LOCATOR.as_posix(),
            "present_and_valid": bool(protocol),
            "error": protocol_error,
            "raw_sha256": (
                _sha256_path(root / MATCH_WINNER_PROTOCOL_LOCATOR)
                if (root / MATCH_WINNER_PROTOCOL_LOCATOR).is_file()
                else None
            ),
            "artifact_sha256": protocol.get("artifact_sha256"),
            "locked_at_utc": protocol.get("locked_at_utc"),
            "result_state": protocol.get("result_state"),
            "phase_one": phase_one,
            "phase_two": phase_two,
            "quote_capture_contract_sha256": protocol.get(
                "quote_capture_contract_sha256"
            ),
            "settlement_contract_sha256": protocol.get(
                "settlement_contract_sha256"
            ),
            "registries": registries,
        },
        "public_bookmaker_terms_snapshot": {
            "locator": BETANO_TERMS_SNAPSHOT_LOCATOR.as_posix(),
            "present_and_valid": bool(public_terms_snapshot),
            "error": public_terms_snapshot_error,
            "raw_sha256": (
                _sha256_path(root / BETANO_TERMS_SNAPSHOT_LOCATOR)
                if (root / BETANO_TERMS_SNAPSHOT_LOCATOR).is_file()
                else None
            ),
            "artifact_sha256": public_terms_snapshot.get("artifact_sha256"),
            "locked_at_utc": public_terms_snapshot.get("locked_at_utc"),
            "coverage": public_terms_snapshot.get("coverage"),
            "authority": public_terms_snapshot.get("authority"),
        },
        "bookmaker_terms_authority": {
            "locator": terms_authority_locator.as_posix(),
            "present_and_valid": bool(terms_authority),
            "error": terms_authority_error,
            "external_digest_pin_present": bool(terms_authority_digest),
            "bookmaker_terms_snapshot_independently_registered": (
                terms_authority.get(
                    "bookmaker_terms_snapshot_independently_registered"
                )
            ),
            "settlement_contract_resolved": terms_authority.get(
                "settlement_contract_resolved"
            ),
            "phase_two_opening_authorized": terms_authority.get(
                "phase_two_opening_authorized"
            ),
        },
        "source_specific_quote_adapter": {
            "candidate": {
                "locator": BETANO_QUOTE_ADAPTER_CANDIDATE_LOCATOR.as_posix(),
                "present_and_valid": bool(quote_adapter_candidate),
                "error": quote_adapter_candidate_error,
                "raw_sha256": quote_adapter_candidate_record.get("raw_sha256"),
                "artifact_sha256": quote_adapter_candidate.get(
                    "artifact_sha256"
                ),
                "locked_at_utc": quote_adapter_candidate.get("locked_at_utc"),
                "result_state": quote_adapter_candidate.get("result_state"),
                "registration": quote_adapter_candidate.get("registration"),
                "live_capture": quote_adapter_candidate.get("live_capture"),
                "authority": quote_adapter_candidate.get("authority"),
            },
            "independent_registry": {
                "locator": REGISTRATIONS["match_winner_quote_adapter"][0].as_posix(),
                "present_and_valid": bool(quote_adapter_registry),
                "error": quote_adapter_registry_error,
                "registry_sha256": quote_adapter_registry.get("registry_sha256"),
                "independent_reviewer_id": quote_adapter_registry.get(
                    "independent_reviewer_id"
                ),
                "authority": quote_adapter_registry.get("authority"),
            },
        },
        "phase_one_collection": {
            **phase_one_collection,
            "registered_readiness": {
                "locator": PHASE_ONE_READINESS_LOCATOR.as_posix(),
                "present_and_valid": bool(collection_readiness),
                "error": collection_readiness_error,
                "raw_sha256": (
                    _sha256_path(root / PHASE_ONE_READINESS_LOCATOR)
                    if (root / PHASE_ONE_READINESS_LOCATOR).is_file()
                    else None
                ),
                "artifact_sha256": collection_readiness.get("artifact_sha256"),
                "locked_at_utc": collection_readiness.get("locked_at_utc"),
                "result_state": collection_readiness.get("result_state"),
                "locked_empty_collection_state": collection_readiness.get(
                    "locked_empty_collection_state"
                ),
                "implementation": collection_readiness.get("implementation"),
                "authority": collection_readiness.get("authority"),
            },
        },
        "phase_one_evaluation": {
            "registered_readiness": {
                "locator": PHASE_ONE_EVALUATION_READINESS_LOCATOR.as_posix(),
                "present_and_valid": bool(evaluation_readiness),
                "error": evaluation_readiness_error,
                "raw_sha256": (
                    _sha256_path(root / PHASE_ONE_EVALUATION_READINESS_LOCATOR)
                    if (root / PHASE_ONE_EVALUATION_READINESS_LOCATOR).is_file()
                    else None
                ),
                "artifact_sha256": evaluation_readiness.get("artifact_sha256"),
                "locked_at_utc": evaluation_readiness.get("locked_at_utc"),
                "result_state": evaluation_readiness.get("result_state"),
                "locked_empty_state": evaluation_readiness.get(
                    "locked_empty_state"
                ),
                "authority": evaluation_readiness.get("authority"),
            },
            "independent_registry": {
                "locator": PHASE_ONE_EVALUATION_REGISTRY_LOCATOR.as_posix(),
                "present_and_valid": bool(evaluation_registry),
                "error": evaluation_registry_error,
                "external_digest_pin_present": bool(
                    evaluation_registry_digest
                ),
                "phase_one_models_independently_passed": evaluation_registry.get(
                    "phase_one_models_independently_passed"
                ),
                "phase_two_opening_authorized": evaluation_registry.get(
                    "phase_two_opening_authorized"
                ),
            },
        },
        "post_pass_probability_pipeline": {
            "registered_readiness": {
                "locator": PROBABILITY_PIPELINE_READINESS_LOCATOR.as_posix(),
                "present_and_valid": bool(probability_pipeline_readiness),
                "error": probability_pipeline_readiness_error,
                "raw_sha256": (
                    _sha256_path(root / PROBABILITY_PIPELINE_READINESS_LOCATOR)
                    if (root / PROBABILITY_PIPELINE_READINESS_LOCATOR).is_file()
                    else None
                ),
                "artifact_sha256": probability_pipeline_readiness.get(
                    "artifact_sha256"
                ),
                "locked_at_utc": probability_pipeline_readiness.get(
                    "locked_at_utc"
                ),
                "result_state": probability_pipeline_readiness.get(
                    "result_state"
                ),
                "runtime_identity": probability_pipeline_readiness.get(
                    "runtime_identity"
                ),
                "locked_empty_state": probability_pipeline_readiness.get(
                    "locked_empty_state"
                ),
                "fresh_post_validation_rating_refit": (
                    probability_pipeline_readiness.get(
                        "probability_pipeline_contract", {}
                    ).get("fresh_post_validation_rating_refit")
                ),
                "authority": probability_pipeline_readiness.get("authority"),
            },
            "independent_registry": {
                "locator": calibration_registry_locator.as_posix(),
                "present_and_valid": bool(calibration_registry),
                "error": calibration_registry_error,
                "external_digest_pin_present": bool(
                    calibration_registry_digest
                ),
                "recalibration_independently_registered": (
                    calibration_registry.get(
                        "recalibration_independently_registered"
                    )
                ),
                "uncertainty_implementation_independently_registered": (
                    calibration_registry.get(
                        "uncertainty_implementation_independently_registered"
                    )
                ),
                "phase_two_opening_authorized": calibration_registry.get(
                    "phase_two_opening_authorized"
                ),
            },
        },
        "event_probability_registry": {
            "locator": probability_registry_locator.as_posix(),
            "present_and_valid": bool(probability_registry),
            "error": probability_registry_error,
            "external_digest_pin_present": bool(probability_registry_digest),
            "registered_receipts": probability_registry.get(
                "registered_receipts"
            ),
            "event_probability_identity_authority": probability_registry.get(
                "event_probability_identity_authority"
            ),
            "probability_accuracy_authorized": probability_registry.get(
                "probability_accuracy_authorized"
            ),
            "betting_authorized": probability_registry.get(
                "betting_authorized"
            ),
        },
        "qualified_quote_registry": {
            "locator": quote_registry_locator.as_posix(),
            "present_and_valid": bool(quote_registry),
            "error": quote_registry_error,
            "external_digest_pin_present": bool(quote_registry_digest),
            "registered_quotes": quote_registry.get("registered_quotes"),
            "quote_identity_authority": quote_registry.get(
                "quote_identity_authority"
            ),
            "odds_accuracy_authorized": quote_registry.get(
                "odds_accuracy_authorized"
            ),
            "betting_authorized": quote_registry.get("betting_authorized"),
        },
        "phase_two_collection_readiness": {
            "locator": PHASE_TWO_COLLECTION_READINESS_REGISTRY_LOCATOR.as_posix(),
            "present_and_valid": bool(phase_two_collection_readiness),
            "error": phase_two_collection_readiness_error,
            "external_digest_pin_present": bool(collection_readiness_digest),
            "independently_registered": phase_two_collection_readiness.get(
                "phase_two_collection_readiness_independently_registered"
            ),
            "betting_authorized": phase_two_collection_readiness.get(
                "betting_authorized"
            ),
        },
        "phase_two_evaluation_readiness": {
            "locator": PHASE_TWO_EVALUATION_READINESS_REGISTRY_LOCATOR.as_posix(),
            "present_and_valid": bool(phase_two_evaluation_readiness),
            "error": phase_two_evaluation_readiness_error,
            "external_digest_pin_present": bool(evaluation_readiness_digest),
            "independently_registered": phase_two_evaluation_readiness.get(
                "phase_two_evaluation_readiness_independently_registered"
            ),
            "betting_authorized": phase_two_evaluation_readiness.get(
                "betting_authorized"
            ),
        },
        "phase_two_stopping_snapshot": {
            "locator": PHASE_TWO_SNAPSHOT_REGISTRY_LOCATOR.as_posix(),
            "present_and_valid": bool(phase_two_snapshot_registry),
            "error": phase_two_snapshot_registry_error,
            "external_digest_pin_present": bool(snapshot_registry_digest),
            "snapshot_identity_authority": phase_two_snapshot_registry.get(
                "phase_two_snapshot_identity_authority"
            ),
            "outcome_opening_authorized": phase_two_snapshot_registry.get(
                "phase_two_outcome_opening_authorized"
            ),
            "betting_authorized": phase_two_snapshot_registry.get(
                "betting_authorized"
            ),
        },
        "phase_two_evaluation": {
            "locator": PHASE_TWO_EVALUATION_REGISTRY_LOCATOR.as_posix(),
            "present_and_valid": bool(phase_two_evaluation_registry),
            "error": phase_two_evaluation_registry_error,
            "external_digest_pin_present": bool(phase_two_evaluation_digest),
            "independently_registered": phase_two_evaluation_registry.get(
                "phase_two_evaluation_independently_registered"
            ),
            "market_gates_independently_passed": phase_two_evaluation_registry.get(
                "phase_two_market_gates_independently_passed"
            ),
            "probability_authorized": phase_two_evaluation_registry.get(
                "probability_authorized"
            ),
            "betting_authorized": phase_two_evaluation_registry.get(
                "betting_authorized"
            ),
        },
        "semantic_market_authority": {
            "locator": SEMANTIC_MATCH_WINNER_AUTHORITY_LOCATOR.as_posix(),
            "present_and_valid": bool(semantic_market_authority),
            "error": semantic_market_authority_error,
            "external_digest_pin_present": authority_pinned,
            "authority_id": (
                semantic_market_authority.get("receipt") or {}
            ).get("authority_id"),
            "private_probability_generation_authorized": (
                semantic_market_authority.get(
                    "private_probability_generation_authorized"
                )
            ),
            "private_decision_support_authorized": (
                semantic_market_authority.get(
                    "private_decision_support_authorized"
                )
            ),
            "transaction_authorized": semantic_market_authority.get(
                "transaction_authorized"
            ),
            "stake_authorized": semantic_market_authority.get(
                "stake_authorized"
            ),
        },
        "phase_two_opening": {
            "locator": REGISTRATIONS[
                "match_winner_phase_two_opening"
            ][0].as_posix(),
            "active_and_valid": bool(phase_two_opening),
            "error": phase_two_opening_error,
            "external_digest_pin_present": bool(
                environment.get(
                    REGISTRATIONS["match_winner_phase_two_opening"][1]
                )
            ),
            "outcome_free_phase_two_collection_active": (
                phase_two_opening.get(
                    "outcome_free_phase_two_collection_active"
                )
            ),
            "probability_authorized": phase_two_opening.get(
                "probability_authorized"
            ),
            "betting_authorized": phase_two_opening.get(
                "betting_authorized"
            ),
        },
        "checks": checks,
        "blockers": blockers,
    }


def _registration_inventory(
    root: Path, environment: Mapping[str, str]
) -> dict[str, Any]:
    return {
        name: {
            "locator": locator.as_posix(),
            "present": (root / locator).is_file(),
            "external_digest_pin_present": bool(environment.get(env_name)),
            "validated": False,
            "validation_note": "requires exact event, league, market, time, and artifact bindings",
        }
        for name, (locator, env_name) in REGISTRATIONS.items()
    }


def inspect_private_decision_readiness(
    root: Path | str = Path("."),
    *,
    as_of: datetime | None = None,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Audit current private readiness without granting event authority."""
    repo_root = Path(root)
    now = as_of or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("as_of must include a timezone")
    now = now.astimezone(timezone.utc)
    env = environment if environment is not None else os.environ
    try:
        draft = inspect_l2_readiness(repo_root, environment=env, as_of=now)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        draft = {
            "status": "invalid",
            "promotion_eligible": False,
            "public_probability_authorized": False,
            "blockers": ["draft_l2_readiness_audit_failed"],
            "error": str(exc),
        }
    ratings = _rating_readiness(repo_root, env, now)
    live_totals = _live_totals_readiness(repo_root, now, env)
    match_winner_market = _match_winner_market_readiness(repo_root, env, now)
    registrations = _registration_inventory(repo_root, env)
    blockers = sorted(
        {
            *[f"draft_score:{item}" for item in draft.get("blockers", [])],
            *[f"ratings:{item}" for item in ratings["blockers"]],
            *[f"live_totals:{item}" for item in live_totals["blockers"]],
            *[
                f"match_winner_market:{item}"
                for item in match_winner_market["blockers"]
            ],
        }
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "as_of": now.isoformat(),
        "status": "blocked" if blockers else "system_evidence_ready",
        "betting_ready": False,
        "event_authorization": {
            "status": "requires_exact_event_replay",
            "self_authorizing": False,
            "note": (
                "This system audit cannot authorize an event. Exact roster, rating, "
                "quote, model, settlement, freshness, and market-authority bindings "
                "must replay in the score request."
            ),
        },
        "draft_score": draft,
        "ratings": ratings,
        "live_totals": live_totals,
        "match_winner_market": match_winner_market,
        "registrations": registrations,
        "blockers": blockers,
    }


__all__ = ["SCHEMA_VERSION", "inspect_private_decision_readiness"]
